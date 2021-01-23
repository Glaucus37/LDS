#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


void init_particles();
void run_sim();
void calculate_forces();
void rand_velocities();
void rand_accel();
void verlet();
double pbc(x);

void aloq();

double* x, * y, * vx, * vy, * ax, * ay;
double dt, dt_sq, sigma;
int N, Lx, Ly;
double t = 0.;
double t_max = 1.;
int step = 0;

#define PI acos(-1.0)


int main() {
	srand((unsigned int)time(NULL));

	N = 10;
	Lx = Ly = 10;
	dt = 0.01;
	dt_sq = dt * dt;
	sigma = 0.1;

	aloq();

	init_particles();
	rand_velocities();

	run_sim();

	return 0;
}



void init_particles() {
	printf("Initial positions:\n");
	for (int i = 0; i < N; i++) {
		x[i] = Lx * (float)rand() / RAND_MAX;
		y[i] = Ly * (float)rand() / RAND_MAX;
		printf("x: %lf\t\ty: %lf\n", x[i], y[i]);
	}
	printf("\n");
}


void run_sim() {
	printf("%d", step);
	do {
		rand_accel();
		verlet();

		step++;
		t += dt * step;

		printf("%d", step);
	} while (t < t_max);
}


void aloq() {
	x = (double*)calloc(N, sizeof(double));
	y = (double*)calloc(N, sizeof(double));

	vx = (double*)calloc(N, sizeof(double));
	vy = (double*)calloc(N, sizeof(double));

	ax = (double*)calloc(N, sizeof(double));
	ay = (double*)calloc(N, sizeof(double));
}


void rand_velocities() {
	double v = 6.;
	double vx_cm = 0;
	double vy_cm = 0;
	double theta;

	//generate random velocities
	for (int i = 0; i < N; i++) {
		theta = 2 * PI * (float)rand() / RAND_MAX;
		vx[i] = v * cos(theta);
		vy[i] = v * sin(theta);
		vx_cm += vx[i];
		vy_cm += vy[i];
	}

	vx_cm /= N;
	vy_cm /= N;

	//correct velocities to set velocity of center of mass to cero.
	for (int i = 0; i < N; i++) {
		vx[i] -= vx_cm;
		vy[i] -= vy_cm;
	}
}


void rand_accel() {
	double a = 5.;
	double ax_cm = 0;
	double ay_cm = 0;
	double theta;

	for (int i = 0; i < N; i++) {
		theta = 2 * PI * (float)rand() / RAND_MAX;
		ax[i] = a * cos(theta);
		ay[i] = a * sin(theta);
		ax_cm += ax[i];
		ay_cm += ay[i];
	}

	ax_cm /= N;
	ay_cm /= N;

	for (int i = 0; i < N; i++) {
		ax[i] -= ax_cm;
		ay[i] -= ay_cm;
	}
}


void verlet() {
	double x_new, y_new;
	double ax_new, ay_new;

	printf("Positions at step %d:\n", step + 1);

	for (int i = 0; i < N; i++) {
		x_new = x[i] + vx[i] * dt + 0.5 * ax[i] * dt_sq;
		y_new = y[i] + vy[i] * dt + 0.5 * ay[i] * dt_sq;
		vx[i] += (ax[i] - sigma * vx[i]) * dt;
		vy[i] += (ay[i] - sigma * vy[i]) * dt;
		x[i] = pbc(x_new);
		y[i] = pbc(y_new);

		printf("x: %lf\t\ty: %lf\n", x[i], y[i]);
	}
}


double pbc(double x) {
	if (x < 0) {
		x += Lx;
	}
	else if (x > Lx) {
		x -= Lx;
	}
	return x;
}
