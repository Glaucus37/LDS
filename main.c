#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
// #include <Python.h>
// #include <numpy/arrayobject.h>


void init_particles();
void run_sim();
void rand_velocities();
void rand_accel();
void verlet();
void vel_half_step();
void verlet_end();
void accel();
double pbc(double x);
double rms();
double kin_energy();
double gauss(double std_dev); // standard deviation
void aloq();
void find_neighbors();

double* x, * y, * vx, * vy, * ax, * ay, * kin_U, * gauss_vel, * k_list;
int** k_neighbors;
double dt, dt_sq, o_sqrt_dt, sigma_;
int N, cells, L;
double t = 0.;
double t_max = 1.5;
double gamma_ = 1.;
int steps = 0;
int steps_max;

// PyObject *pName, *pModule;

#define PI acos(-1.0)
#define lat_size 4
#define dt 0.01

int main(){
	/*
	PyInitialize();
	pName = PyUnicode_FromString(plots.py);
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);
	*/
	srand((unsigned int)time(NULL));

	N = 10; // number of particles in system
	L = 10; // side length

	//defining constants
	dt_sq = dt * dt;
	o_sqrt_dt = 1 / sqrt(dt);
	double kBT = 1;
	double m = 1;
	sigma_ = o_sqrt_dt * sqrt(2 * gamma_ * kBT * m);

	steps_max = (int)(t_max / dt) + 1; //calculate number of steps to be taken

	cells = (int)(lat_size * lat_size);

	aloq(); // allocate memory for global variables/arrays
	find_neighbors();

	init_particles(); // initialize particles at random positions
	rand_velocities(); // set random velocities
	rand_accel(); //set initial random acceleration

	run_sim(); // run the actual iterative simulation

	printf("\nAverage velocity (RMS): %.3lf m/s\n", rms());

	FILE *fp;
	fp = fopen("energy.txt", "w");
	for(int i = 0; i < steps_max; i++){
		fprintf(fp, "%.6lf\n", kin_U[i]);
	}
	fclose(fp);

	return 0;
}


void run_sim(){
	do {
		verlet(); // recalculate position and velocity
		vel_half_step(); // half-step update of velocity
		accel(); // calculate acceleration on particles
		vel_half_step(); // second half_step update
		kin_U[steps] = kin_energy(); //keeping track of system's energy

		t += dt; // keeping track of time
		steps++;
	} while (t < t_max);
}


void init_particles(){
	for(int i = 0; i < N; i++){
		x[i] = L * (float)rand() / RAND_MAX; //random number between 0 and 1
		y[i] = L * (float)rand() / RAND_MAX;
	}
}


void aloq(){
	x = (double*)calloc(N, sizeof(double));
	y = (double*)calloc(N, sizeof(double));

	vx = (double*)calloc(N, sizeof(double));
	vy = (double*)calloc(N, sizeof(double));

	ax = (double*)calloc(N, sizeof(double));
	ay = (double*)calloc(N, sizeof(double));

	k_neighbors = (int**)calloc(cells, sizeof(int*));
	for(int i = 0; i < cells; i++){
		k_neighbors[i] = (int*)calloc(5, sizeof(int));
	}

	kin_U = (double*)calloc(steps_max, sizeof(double));
	gauss_vel = (double*)calloc(2, sizeof(double));

	//neighbors = (int*)calloc(5, sizeof(int));
	k_list = (int*)calloc((int)(N + cells), sizeof(int));
}


void rand_velocities(){
	double v = 3.; // max speed per dimension
	double vx_cm = 0;
	double vy_cm = 0;
	double theta; //velocity direction

	//generate random velocities
	for(int i = 0; i < N; i++){
		theta = 2 * PI * (float)rand() / RAND_MAX;
		vx[i] = v * cos(theta);
		vy[i] = v * sin(theta);
		vx_cm += vx[i]; // velocity of center of mass
		vy_cm += vy[i];
	}

	vx_cm /= N; // average out the velocities of all the particles
	vy_cm /= N;

	//correct velocities to cancel center of mass drift
	for(int i = 0; i < N; i++){
		vx[i] -= vx_cm;
		vy[i] -= vy_cm;
	}
}


void rand_accel(){ //same principle as with velocity
	double a = 5.;
	double ax_cm = 0;
	double ay_cm = 0;
	double theta;

	for(int i = 0; i < N; i++){
		theta = 2 * PI * (float)rand() / RAND_MAX;
		ax[i] = a * cos(theta);
		ay[i] = a * sin(theta);
		ax_cm += ax[i];
		ay_cm += ay[i];
	}

	ax_cm /= N;
	ay_cm /= N;

	for(int i = 0; i < N; i++){
		ax[i] -= ax_cm;
		ay[i] -= ay_cm;
	}
}


void verlet(){ // update positions based on velocity and acceleration
	double x_new, y_new;
	double ax_new, ay_new;

	for(int i = 0; i < N; i++){
		x_new = x[i] + vx[i] * dt + 0.5 * ax[i] * dt_sq;
		y_new = y[i] + vy[i] * dt + 0.5 * ay[i] * dt_sq;
		x[i] = pbc(x_new);
		y[i] = pbc(y_new);
	}
}


void vel_half_step(){ // update velocity at half-step intervals
	for(int i = 0; i < N; i++){
		vx[i] += 0.5 * ax[i] * dt;
		vy[i] += 0.5 * ay[i] * dt;
	}
}


/*
* Note about accel():
* I added and initial run of gauss() to be able to call it again in
* the middle of every step. This is to delete any possible correlation
* between vx and xy for every particle at every step.
*
* @Glaucus37
*/
void accel(){ // update acceleration based on...
	gauss(1.);
	for(int i = 0; i < N; i++){
						// ...drag				// ...and browninan motion
		ax[i] = -gamma_ * vx[i] + sigma_ * gauss_vel[0];
		gauss(1.);
		ay[i] = -gamma_ * vy[i] + sigma_ * gauss_vel[1];
	}
}


// Periodic Boundary Conditions
double pbc(double x){ // x (or y)
	if (x < 0){
		x += L;
	}
	else if (x > L){
		x -= L;
	}
	return x;
}


double rms(){ // root medium square
	double ms = 0;

	for(int i = 0; i < N; i++){
		ms += vx[i] * vx[i] + vy[i] * vy[i];
	}
	return sqrt(ms / N);
}


double kin_energy(){ // calculate kinetic energy of the whole system
	double kin = 0;
	for(int i = 0; i < N; i++){
		kin += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]) * dt_sq;
	}

	return kin;
}


double gauss(double std_dev){ // generate pair of numbers with given dev
	double r_sq, fac, v1, v2;
	do{
		v1 = 2. * rand() / RAND_MAX - 1;
		v2 = 2. * rand() / RAND_MAX - 1;
		r_sq = v1 * v1 + v2 * v2;
	} while(r_sq > 1. || r_sq == 0);
	fac = std_dev * sqrt(-2. * log(r_sq) / r_sq);
	gauss_vel[0] = v1 * fac;
	gauss_vel[1] = v2 * fac;
}


void find_neighbors(){
	// "naive" neighbors, aka disregarding pbc
	static int naive_neighbors[5] = {0, 1, lat_size - 1, lat_size, lat_size + 1};
	int neighbors[5]; // local array to calculate neighbors

	for(int k = 0; k < cells; k++){
		// initialize neighbors at naive_neighbors
		memcpy(neighbors, naive_neighbors, sizeof(naive_neighbors));
		for(int i = 0; i < 5; i++){
			neighbors[i] += k;
		}

		if(k % lat_size == 0){
			neighbors[2] += lat_size;
		}else if(k % lat_size == lat_size - 1){
			neighbors[1] -= lat_size;
			neighbors[4] -= lat_size;
		}
		if(floor(k / lat_size) == lat_size - 1){
			neighbors[2] -= cells;
			neighbors[3] -= cells;
			neighbors[4] -= cells;
		}
	}
}
