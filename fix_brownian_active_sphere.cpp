/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   //*********************************************************************************
   // algorithm:
   /*********************************************************************************
   // director(t+dt) = Rotation(omega)*director(t)
   // v(t+dt)        = invDamp*force(t) + sqrt(2*D/dt)*noise + activeVel*director(t+dt)
   // r(t+dt)        = r(t) + dt*v(t+dt)
   // omega(t+dt)    = invDampRot*Torque(t) + sqrt(2*Dr/dt)*noise

   // Dimension: [invDamp]    = [Time]/[Mass]
   // Dimension: [invDampRot] = [Time]/[Mass]/[Length^2]
   // Dimension: [D]          = [Length^2]/[Time]
   // Dimension: [Dr]         = 1/[Time]
   // dt integration time

   // According to Brownian dynamics and Einstein relation 
   // the two following relations are always true irrespective of Dimension:
   // 1. D           = 4*rad^2*Dr/3
   // 2. D*invDamp   = Dr*invDampRot
   //*********************************************************************************

------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fix_brownian_active_sphere.h"
#include "atom.h"
#include "domain.h"
#include "atom_vec.h"
#include "update.h"
#include "comm.h"
#include "respa.h"
#include "force.h"
#include "error.h"
#include "math_vector.h"
#include "math_extra.h"
#include "random_mars.h"
#include <sstream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathExtra;

enum{NONE,DIPOLE};
enum{NODLM,DLM};

/* ---------------------------------------------------------------------- */

FixBrownianActiveSphere::FixBrownianActiveSphere(LAMMPS *lmp, int narg, char **arg) :
  FixNVE(lmp, narg, arg), random(NULL)
{
      //if (narg < 3) error->all(FLERR,"Illegal fix brownian/active/sphere command");
      if (narg < 7) error->all(FLERR,"Illegal fix brownian/active/sphere command");

  time_integrate = 1;

  // process extra keywords
  // inertia = moment of inertia prefactor for sphere or disc

  extra = NONE;
  dlm = NODLM;
  inertia = 0.4;

  /*******************************************************************************/
  Damp      = atof(arg[3]);
  activeMag = atof(arg[4]);
  noiseMag  = atof(arg[5]);
  seed      = atoi(arg[6]);
  // initialize Marsaglia RNG with processor-unique seed
  random    = new RanMars(lmp,seed + comm->me);

  zDimTrue  = 1;
  if (domain->dimension == 2) zDimTrue = 0;

 
  /*******************************************************************************/


  int iarg = 7;
  // int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"update") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix brownian/active/sphere command");
      if (strcmp(arg[iarg+1],"dipole") == 0) extra = DIPOLE;
      else if (strcmp(arg[iarg+1],"dipole/dlm") == 0) {
        extra = DIPOLE;
        dlm = DLM;
      } else error->all(FLERR,"Illegal fix brownian/active/sphere command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"disc")==0) {
      inertia = 0.5;
      if (domain->dimension != 2)
	error->all(FLERR,"Fix brownian/active/sphere disc requires 2d simulation");
      iarg++;
    }
    else if (strcmp(arg[iarg],"nvz")==0) { // velocity along z = 0
          zDimTrue = 0;
          iarg++;
    }
    else error->all(FLERR,"Illegal fix brownian/active/sphere command");
  }

  // error checks

  if (!atom->sphere_flag)
    error->all(FLERR,"Fix brownian/active/sphere requires atom style sphere");
  if (extra == DIPOLE && !atom->mu_flag)
    error->all(FLERR,"Fix brownian/active/sphere update dipole requires atom attribute mu");

  std::stringstream ssrm;
  ssrm << "# damp, activeMAG, noiseMag, seed, zDimension: " << Damp << "\t" << activeMag << "\t"
       << noiseMag << "\t" << seed << "\t" << zDimTrue << "\n";
  error->message(FLERR,ssrm.str().c_str());
  ssrm.clear();
}

/* ---------------------------------------------------------------------- */

void FixBrownianActiveSphere::init()
{
  FixNVE::init();

  // check that all particles are finite-size spheres
  // no point particles allowed

  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (radius[i] == 0.0)
        error->one(FLERR,"Fix brownian/active/sphere requires extended particles");
}

/* ---------------------------------------------------------------------- */

void FixBrownianActiveSphere::initial_integrate(int vflag)
{
  double dtfm,dtirotate,msq,scale,s2,inv_len_mu;
  double g[3];
  vector w, w_temp, a;
  matrix Q, Q_temp, R;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  /**************************************************************/
  // reading director information
  // set using fix property/atom
  // first column contains azimuthal angle of sphere i [-pi to pi]
  // second column contains polar angle of sphere i [-pi to pi]

  double **director = atom->dvector;
  double oneByDamp, oneByDampTorque , vnoisePref, rnoisePref;
  double r3,r5;
  double dirx, diry, dirz, dirMag, ox, oy, oz;
  /**************************************************************/

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set timestep here since dt may have changed or come via rRESPA

  // double dtfrotate  = dtf / inertia;
  // double ForcePref  = 0.5/Damp;
  // double torquePref = 0.375/(Damp*inertia);

  /******************************************************/
  // let, psi(n) = 2An, n is a random number [-0.5,0.5)
  // probability: P(n) = 4A^2 \int_{-0.5}^{0.5} n^2 dn =1
  // A = sqrt(3)
  // psi(n) = 2*sqrt(3)*n
  double noisePref  = sqrt(12.0*noiseMag/dtv); // noiseMag = 2*Dr
  /*****************************************************/


  // update v,x,omega for all particles
  // d_omega/dt = torque / inertia

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
          // dtfm = dtf / rmass[i];
          // v[i][0] += dtfm * f[i][0];
          // v[i][1] += dtfm * f[i][1];
          // v[i][2] += dtfm * f[i][2];

          /***/
          // std::stringstream ssrm;
          // ssrm << "### 1: " << i << "\t" << rmass[i] << "\t" << director[0][i] << "\t"
          //       << director[1][i] << "\t" << director[2][i] << "\t" << dirMag << "\n"
          //      << v[i][0] << "\t" << v[i][1] << "\t" << v[i][2] << "\t"
          //      << v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2] << "\n"
          //      << omega[i][0] << "\t" << omega[i][1] << "\t" << omega[i][2] << "\t"
          //      << omega[i][0]*omega[i][0]+omega[i][1]*omega[i][1]+omega[i][2]*omega[i][2] << "\n" ;
          //      << dtv << "\n";
          // error->message(FLERR,ssrm.str().c_str());
          // ssrm.clear();
          /***/

          /***************************************************************/
          // update directors
          ox = dtv * omega[i][0];
          oy = dtv * omega[i][1];
          oz = dtv * omega[i][2];

          // update angular displacement (REQUIRED FOR ROTATIONAL MSD)
          director[3][i] += ox;
          director[4][i] += oy;
          director[5][i] += oz;

          // rotate around x
          dirx = director[0][i];
          diry = director[1][i]*cos(ox)-director[2][i]*sin(ox);
          dirz = director[1][i]*sin(ox)+director[2][i]*cos(ox);
          director[0][i] = dirx;
          director[1][i] = diry;
          director[2][i] = dirz;

          // rotate around y
          dirx = director[0][i]*cos(oy)+director[2][i]*sin(oy);
          diry = director[1][i];
          dirz =-director[0][i]*sin(oy)+director[2][i]*cos(oy);
          director[0][i] = dirx;
          director[1][i] = diry;
          director[2][i] = dirz;

          // rotate around z
          dirx = director[0][i]*cos(oz)-director[1][i]*sin(oz);
          diry = director[0][i]*sin(oz)+director[1][i]*cos(oz);
          dirz = director[2][i];
          director[0][i] = dirx;
          director[1][i] = diry;
          director[2][i] = dirz;

          dirMag = sqrt(director[0][i]*director[0][i] + director[1][i]*director[1][i]
                        +director[2][i]*director[2][i]);

          director[0][i] /= dirMag;
          director[1][i] /= dirMag;
          director[2][i] /= dirMag;
          /***************************************************************/

          oneByDamp  = 1.0/(Damp*rmass[i]);         // ForcePref/(radius[i]*rmass[i]);
          vnoisePref = noisePref*sqrt(4.0*radius[i]*radius[i]/3.0); // noisePref/sqrt(6*radius[i]);

          v[i][0] = oneByDamp*f[i][0] + vnoisePref*(random->uniform()-0.5) + activeMag*director[0][i];
          v[i][1] = oneByDamp*f[i][1] + vnoisePref*(random->uniform()-0.5) + activeMag*director[1][i];
          v[i][2] = oneByDamp*f[i][2] + vnoisePref*(random->uniform()-0.5) + activeMag*director[2][i];
          v[i][2] = zDimTrue*v[i][2]; // =0 if two dimension
          
          x[i][0] += dtv * v[i][0];
          x[i][1] += dtv * v[i][1];
          x[i][2] += dtv * v[i][2];

          //r3= radius[i]*radius[i]*radius[i];
          //r5= r3*radius[i]*radius[i];

          oneByDampTorque = 0.75/(Damp*rmass[i]*radius[i]*radius[i]); //torquePref/(rmass[i]*r5);
          //rnoisePref    = noisePref/sqrt(8*r3);

          omega[i][0] = oneByDampTorque*torque[i][0] + noisePref*(random->uniform()-0.5); //rnoisePref*(random->uniform()-0.5);
          omega[i][1] = oneByDampTorque*torque[i][1] + noisePref*(random->uniform()-0.5); //rnoisePref*(random->uniform()-0.5);
          omega[i][2] = oneByDampTorque*torque[i][2] + noisePref*(random->uniform()-0.5); //rnoisePref*(random->uniform()-0.5);

          /***/
          // std::stringstream ssrm2;
          // ssrm2 << "### 2: " << i << "\t" << rmass[i] << "\t" << director[0][i] << "\t"
          //      << director[1][i] << "\t" << director[2][i] << "\t" << dirMag << "\n"
          //      << v[i][0] << "\t" << v[i][1] << "\t" << v[i][2] << "\t"
          //      << v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2] << "\n"
          //      << omega[i][0] << "\t" << omega[i][1] << "\t" << omega[i][2] << "\t"
          //      << omega[i][0]*omega[i][0]+omega[i][1]*omega[i][1]+omega[i][2]*omega[i][2] << "\n";
          // error->message(FLERR,ssrm2.str().c_str());
          // ssrm2.clear();
          /***/
    }
  }

  // update mu for dipoles

  if (extra == DIPOLE) {
    double **mu = atom->mu;
    if (dlm == NODLM) {

      // d_mu/dt = omega cross mu
      // renormalize mu to dipole length

      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          if (mu[i][3] > 0.0) {
            g[0] = mu[i][0] + dtv * (omega[i][1]*mu[i][2]-omega[i][2]*mu[i][1]);
            g[1] = mu[i][1] + dtv * (omega[i][2]*mu[i][0]-omega[i][0]*mu[i][2]);
            g[2] = mu[i][2] + dtv * (omega[i][0]*mu[i][1]-omega[i][1]*mu[i][0]);
            msq = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
            scale = mu[i][3]/sqrt(msq);
            mu[i][0] = g[0]*scale;
            mu[i][1] = g[1]*scale;
            mu[i][2] = g[2]*scale;
          }
    } else {

      // integrate orientation following Dullweber-Leimkuhler-Maclachlan scheme

      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit && mu[i][3] > 0.0) {

          // Construct Q from dipole:
          // Q is the rotation matrix from space frame to body frame
          // i.e. v_b = Q.v_s

          // define mu to lie along the z axis in the body frame
          // take the unit dipole to avoid getting a scaling matrix

          inv_len_mu = 1.0/mu[i][3];
          a[0] = mu[i][0]*inv_len_mu;
          a[1] = mu[i][1]*inv_len_mu;
          a[2] = mu[i][2]*inv_len_mu;

          // v = a x [0 0 1] - cross product of mu in space and body frames
          // s = |v|
          // c = a.[0 0 1] = a[2]
          // vx = [ 0    -v[2]  v[1]
          //        v[2]  0    -v[0]
          //       -v[1]  v[0]  0    ]
          // then
          // Q = I + vx + vx^2 * (1-c)/s^2

          s2 = a[0]*a[0] + a[1]*a[1];
          if (s2 != 0.0){ // i.e. the vectors are not parallel
            scale = (1.0 - a[2])/s2;

            Q[0][0] = 1.0 - scale*a[0]*a[0];
            Q[0][1] = -scale*a[0]*a[1];
            Q[0][2] = -a[0];
            Q[1][0] = -scale*a[0]*a[1];
            Q[1][1] = 1.0 - scale*a[1]*a[1];
            Q[1][2] = -a[1];
            Q[2][0] = a[0];
            Q[2][1] = a[1];
            Q[2][2] = 1.0 - scale*(a[0]*a[0] + a[1]*a[1]);
          } else { // if parallel then we just have I or -I
            Q[0][0] = 1.0/a[2];  Q[0][1] = 0.0;       Q[0][2] = 0.0;
            Q[1][0] = 0.0;       Q[1][1] = 1.0/a[2];  Q[1][2] = 0.0;
            Q[2][0] = 0.0;       Q[2][1] = 0.0;       Q[2][2] = 1.0/a[2];
          }

          // Local copy of this particle's angular velocity (in space frame)
          w[0] = omega[i][0]; w[1] = omega[i][1]; w[2] = omega[i][2];

          // Transform omega into body frame: w_temp= Q.w
          matvec(Q,w,w_temp);

          // Construct rotation R1
          BuildRxMatrix(R, dtf/force->ftm2v*w_temp[0]);

          // Apply R1 to w: w = R.w_temp
          matvec(R,w_temp,w);

          // Apply R1 to Q: Q_temp = R^T.Q
          transpose_times3(R,Q,Q_temp);

          // Construct rotation R2
          BuildRyMatrix(R, dtf/force->ftm2v*w[1]);

          // Apply R2 to w: w_temp = R.w
          matvec(R,w,w_temp);

          // Apply R2 to Q: Q = R^T.Q_temp
          transpose_times3(R,Q_temp,Q);

          // Construct rotation R3
          BuildRzMatrix(R, 2.0*dtf/force->ftm2v*w_temp[2]);

          // Apply R3 to w: w = R.w_temp
          matvec(R,w_temp,w);

          // Apply R3 to Q: Q_temp = R^T.Q
          transpose_times3(R,Q,Q_temp);

          // Construct rotation R4
          BuildRyMatrix(R, dtf/force->ftm2v*w[1]);

          // Apply R4 to w: w_temp = R.w
          matvec(R,w,w_temp);

          // Apply R4 to Q: Q = R^T.Q_temp
          transpose_times3(R,Q_temp,Q);

          // Construct rotation R5
          BuildRxMatrix(R, dtf/force->ftm2v*w_temp[0]);

          // Apply R5 to w: w = R.w_temp
          matvec(R,w_temp,w);

          // Apply R5 to Q: Q_temp = R^T.Q
          transpose_times3(R,Q,Q_temp);

          // Transform w back into space frame w_temp = Q^T.w
          transpose_matvec(Q_temp,w,w_temp);
          omega[i][0] = w_temp[0];
          omega[i][1] = w_temp[1];
          omega[i][2] = w_temp[2];

          // Set dipole according to updated Q: mu = Q^T.[0 0 1] * |mu|
          mu[i][0] = Q_temp[2][0] * mu[i][3];
          mu[i][1] = Q_temp[2][1] * mu[i][3];
          mu[i][2] = Q_temp[2][2] * mu[i][3];
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixBrownianActiveSphere::final_integrate()
{
  double dtfm,dtirotate;

  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  /**************************************************************/
  // reading director information
  // set using fix property/atom
  // first column contains azimuthal angle of sphere i [-pi to pi]
  // second column contains polar angle of sphere i [-pi to pi]

  //double **director = atom->dvector;
  //double oneByDampMass, velx, vely, velz;
  /**************************************************************/

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set timestep here since dt may have changed or come via rRESPA

  // double dtfrotate = dtf / inertia;

  // update v,omega for all particles
  // d_omega/dt = torque / inertia

  double rke = 0.0;
  for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
              //     // dtfm = dtf / rmass[i];
              //     // v[i][0] += dtfm * f[i][0];
              //     // v[i][1] += dtfm * f[i][1];
              //     // v[i][2] += dtfm * f[i][2];

              // dtirotate = dtfrotate / (radius[i]*radius[i]*rmass[i]);
              // omega[i][0] += dtirotate * torque[i][0];
              // omega[i][1] += dtirotate * torque[i][1];
              // omega[i][2] += dtirotate * torque[i][2];
              rke += (omega[i][0]*omega[i][0] + omega[i][1]*omega[i][1] +
                      omega[i][2]*omega[i][2])*radius[i]*radius[i]*rmass[i];
        }

}
