using namespace std;

//asce etsz
class Rad{
public:
  Rad(int YY, int MM, int DD, int HH, double llat, double llonm, double llonz, double SSIn;
~Rad(void);
  int J(void);
  double sc(void);
  double dr(void);
  double dec(void);
  double phi(void);
  double omega(void);
  double omega_s(void);
  double omega_1(void);
  double omega_2(void);
  double beta(void);
  double S(void);
  double fcd(void);
  double fcd_beta_lt_03(void);
  double So(void);
  double Sn(double alb);
  double Ln(void);
  double Rn(void);
  double Rnc(double LAI);
};


class Air{
public:
  Air(double TTa, double RRH, double uuz, double zz, double ZZZ, double hhc, double PPmb);
  ~Air(void);
  double u2(void);
  double delta(void);
  double P(void);
  double gamma(void);
  double e_T(double T);
  double e_s(void);
  double e_a(void);
  double zm(void);
  double zh(void);
  double d(void);
  double ra(void);
  double eps_atm(void);
};

class Canopy{
public:
  Canopy(double fc1, double ff1, double hc1, double Tg1);
  ~Canopy(void);
  double Kb(double thetar);
  double alb(void);
  double omega0(void);
  double omega(double theta);
  double LAI(void);
  double porosity(void);
  double tau_bt(double theta, double alpha);
  double tau_d(void);
  double fveg(void);
  double Kc(void);
  double zm(void);
  double zh(void);
  double d(void);
  double uc(double uz, double z);
  double rc(double uz, double z);
  double T(double Ts, double eps_s, double Trad, double eps_rad);
};

class Soil{
public:
  Soil(double TTs);
  ~Soil(void);
  double us(Canopy can, Air air);
  double rs(Canopy can, Air air);
};

double gamma_star(double, double ra, double rc);
double Tdry(Air air, double Rnc);
double Twet(Air air, Canopy grapes, double Rnc);
double CWSI(double Tcan, double Tdry, double Twet);
double Tsoil(Canopy grapes, Air air, Rad rad, Soil soil, double eps_rad, double Trad);
double Tgrass(Canopy grapes, Air air, Rad rad, Canopy grass, double eps_rad, double Trad, double Tg);
