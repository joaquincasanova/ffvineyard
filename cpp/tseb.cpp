#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

double pi = 3.1459265;
double boltz = 0.00000005670373;// W K-4 m-2
double vonk = 0.4;
double Tk = 273.16;
double gammac = 0.000665;
double cp = 1005; //J/kg/K
double rhoa = 1.205; //kg/m3
 
//asce etsz
class Rad{
public:
  Rad(int, int, int, int, double, double, double, double);
  ~Rad(void);
  int D;
  int M;
  int Y;
  int H;
  double lat;//deg
  double lonm;//deg-longitude of measurement site
  double lonz;//deg
  double SIn;//MJ/m2/h
  int J(void){return D-32+(int)(275*M/9)+2*(int)(3/(M+1))+(int)(M/100-(Y%4)/4+0.975);}
  double sc(void){
    double b=2*pi*(J()-81)/364;
    return 0.1645*sin(2*b)-0.1255*cos(b)-0.025*sin(b);
  }//hour
  double dr(void){return 1+0.033*cos(2*pi/365*J());}
  double dec(void){return 0.409*sin(2*pi/365*J()-1.39);}
  double phi(void){return pi/180*lat;}
  double omega(void){return pi/12*(H+0.06667*(lonz-lonm)+sc()-12);}
  double omega_s(void){return acos(-tan(phi())*tan(dec()));}
  double omega_1(void){
    double tmp=omega()-pi*H/24; 
    if(tmp<-omega_s()){
      tmp=-omega_s();
    }
    if(tmp>omega_s()){
      tmp=omega_s();
    }
    return tmp;
  }
  double omega_2(void){
    double tmp=omega()+pi*H/24;
    if(tmp<-omega_s()){
      tmp=-omega_s();
    }
    if(tmp>omega_s()){
      tmp=omega_s();
    }
    return tmp;
  }
  double beta(void){
    return asin(sin(phi())*sin(dec())+cos(phi())*cos(dec())*cos(omega()));
  }
  double S(void){
    double tmp1 = (omega_2()-min(omega_1(),omega_2()))*sin(phi())*sin(dec());
    double tmp2 = (sin(omega_2()-min(omega_1(),omega_2())))*cos(phi())*cos(dec());
    return 12/pi*4.92*dr()*(tmp1+tmp2)*1000000/3600;
  }//W/m2
  double fcd(void){
    return (1.35*S()/So()-0.35);
  }
  double fcd_beta_lt_03(void){
    int H0 = H;
    H=1;
    while(beta()<0.3){H++;}
    double tmp = (1.35*S()/So()-0.35);
    H = H0;
    return tmp;
  }
  double So(void){return (0.75+(2e-5)*ZZ)*S();}
  double Sn(double alb){
    if(SIn<0){return So()*(1.0-alb);}
    else{return SIn*(1.0-alb);}
  }
  double Ln(void){
    if (beta()>=0.3){
      return boltz*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a()))*fcd();
    }else{
      return boltz*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a()))*fcd_beta_lt_03();
    }
  }
  double Rn(void){return Sn()-Ln();}//W/m2
  double Rnc(double LAI){return Rn*(1-exp(-0.6*LAI/sqrt(2*cos(beta))));}//C&N 99
};

Rad::Rad(int YY, int MM, int DD, int HH, double llat, double llonm, double llonz, double SSIn){
  int D=DD;
  int M=MM;
  int Y=YY;
  int H=HH;
  double lat=llat;//deg
  double lonm=llonm;//deg-longitude of measurement site
  double lonz=llonz;//deg
  double SIn=SSIn;//MJ/m2/h
}

Rad::~Rad(void){
}

class Air{
public:
  Air(double, double, double, double, double,double,double);
  ~Air(void);
  double Ta;//C
  double RH;//%
  double uz;//m/s
  double z;//m
  double ZZ;//m
  double hc;//m
  double Pmb;//mb
  double u2(void){if(z==2){return uz;}else{return uz*4.87/log(67.8*z-5.42);}} //m/s, m
  double delta(void){return 4098*(0.6108*exp(17.27*Ta/(Ta+237.7)))/pow(Ta+273.3,2);} //kpa/C
  double P(void){
    if (Pmb < 0){return 101.3*pow(((293-0.0065*ZZ)/293),5.26);}
    else{return Pmb*0.1;}
  }//kpa, 
  double gamma(void){return P()*gammac;}//kpa/c
  double e_T(double T){return 0.6108*exp(17.27*T/(T+237.3));}
  double e_s(void){return (e_T(Ta));}
  double e_a(void){return (e_T(Ta)*RH/100)/2;}
  double zm(void){return 0.13*hc;}
  double zh(void){return zm()/7;}
  double d(void){return 0.65*hc;}
  double ra(void){return log((z-d())/zm())*log((z-d())/zh())/(vonk*vonk)/rhoa/cp/uz;}//C&N 1998, nutral stability
  double eps_atm(void){return 0.70+0.000595*e_a()*exp(1500/(Ta+Tk));}
};

Air::Air(double TTa, double RRH, double uuz, double zz, double ZZZ, double hhc, double PPmb){
  double Ta = TTa;//C
  double RH = RRH;//%
  double uz = uuz;//m/s
  double z = zz;//m
  double ZZ = zzz;//m
  double hc = hhc;//m
  double Pmb = PPmb;//mb
}

Air::~Air(void){
}

class Canopy{
public:
  Canopy(double,double,double,double);
  ~Canopy(void);
  double Tg;
  double x;//ellipsoidal leaf angle parameter
  double leaf;//leaf size
  double fc;//crown fraction
  double ff;//total fraction
  double ke;//extinction
  double hc;//foliage height m
  double eps_c;//emissivity
  double Kb(double thetar){return sqrt(x*x+pow(tan(thetar),1))/(x+1.774*pow(x+1.182,-0.733));}//Campbell and Norman 98
  double alb(void){return 0.2;}//albedo from 
  double omega0(void){return (1-porosity())*log(1-ff)/log(porosity())/ff;}//Fuentes 2008
  double omega(double theta){
    double D = 1;
    double p = 3.80 - 0.46*D;
    double tmp1 = omega0()+(1-omega0())*exp(-2.2*pow(theta,p))
      return omega0()/tmp1;
  }
  double LAI(void){return -omega0()*fc*log(porosity())/ke;}
  double porosity(void){return ff/fc;}
  double tau_bt(double theta, double alpha){return exp(-Kb(theta)*omega(theta)*LAI()*sqrt(alpha));}
  double tau_d(void){
    double theta=0;
    double n=100;
    double del=pi/2/n;
    double sum=0;
    for(theta=0;theta<=pi/2;theta+=del){
      sum += tau_bt(theta, 1)*del*2;
    }
    return sum;
  }
  double fveg(void){return 1-tau_d;}
  double Kc(void){return 0.115+0.235*LAI();}//Ayars paper 2005
  double zm(void){return 0.13*hc;}
  double zh(void){return zm()/7;}
  double d(void){return 0.65*hc;}
  double uc(double uz, double z){return uz*(log((hc-d())/zm()))/(log((z-d())/zm()));}
  double rc(double uz, double z){
    double C = 90;//sqrt(s)/m grace 1981 via campbell norman 95
    double a = 0.28*pow(LAI,2.0/3.0)*pow(hc/leaf,1.0/3.0);
    double udzm = uc(uz,z)*exp(-a*(1-(d()+zm())/hc));
    return C/LAI()*sqrt(leaf/udzm);}
  double T(double Ts, double eps_s, double Trad, double eps_rad){
    double G = boltz*eps_rad*pow(Trad+Tk,4);
    double F = boltz*eps_s*pow(Ts+Tk,4)*(1-fveg());
    double E = boltz*eps_c*fveg();
    return pow((G-F)/E,0.25)-Tk;}
};

Canopy::Canopy(double fc1, double ff1, double hc1, double Tg1){
  fc=fc1;
  ff=ff1;
  hc=hc1;
  ke=0.7;
  x=2;
  eps_c=0.98;
  Tg=Tg1;
}

Canopy::~Canopy(void){
}
class Soil{
public:
  Soil(double);
  ~Soil(void);
  double Ts;
  double eps_s;
  double us(Canopy can, Air air){
    double a = 0.28*pow(can.LAI(),0.66666)*pow(can.hc/can.leaf,0.333333);
    return can.uc(air.uz,air.z)*exp(-a*(1-0.05/can.hc));
  }
  double rs(Canopy can, Air air){
    double ap = 0.004;
    double bp = 0.012;
    return 1/(ap+bp*us(can, air));}
  double Ls(void){return eps_s*boltz*pow(Ts+Tk,4.0);}
};

Soil::Soil(double TTs){
  Ts=TTs;
  eps_s=0.98;
}

~Soil::Soil(void){
}

double gamma_star(double gamma, double ra, double rc){return gamma*(rc/ra);}//c&n 1998//kpa/c
double Tdry(Air air, double Rnc){return air.Ta+air.ra()*Rnc/(rhoa*cp);}
double Twet(Air air, Canopy grapes, double Rnc){
  double ra = air.ra();
  double rc = grapes.rc(air.uz, air.z);
  double delta = air.delta();
  double gs = gamma_star(air.gamma(), ra, rc);
  double tmp1 = ra*gs*Rnc/(rhoa*cp*(gs+delta));
  double tmp2 = (air.e_s()-air.e_a())/(gs+delta);
  return air.Ta + tmp1 - tmp2;
}
 
double CWSI(double Tcan, double Tdry, double Twet){return (Tcan-Twet)/(Tdry-Twet);}

double Tsoil(Canopy grapes, Air air, Rad rad, Soil soil, double eps_rad, double Trad){
  double ra = air.ra();
  double rc = grapes.rc(air.uz, air.z);
  double LAI = grapes.LAI();
  double rs = soil.rs(grapes, air);
  double delta = air.delta();
  double gs = gamma_star(air.gamma(), ra, rc);
  double A=-rc*gs*rad.Rnc(LAI)/(delta+gs)/rhoa/cp+rc/ra*(air.e_s()-air.e_a())/(delta+gs);
  double tmp = (1/ra+1/rc+1/rs);
  double B=(air.Ta/ra)/tmp;
  double C=(1/rs)/tmp;
  double D=(1/rc)/tmp;
  return (grapes.T(soil.Ts, soil.eps_s, Trad, eps_rad)*(1-D)+A-B)/C;    
}
double Tgrass(Canopy grapes, Air air, Rad rad, Canopy grass, double eps_rad, double Trad, double Tg){
  double ra = air.ra();
  double rc = grapes.rc(air.uz, air.z);
  double rs = grass.rc(air.uz, air.z);
  double LAI = grapes.LAI();
  double delta = air.delta();
  double gs = gamma_star(air.gamma(), ra, rc);
  double A=-rc*gs*rad.Rnc(LAI)/(delta+gs)/rhoa/cp+rc/ra*(air.e_s()-air.e_a())/(delta+gs);
  double tmp = (1/ra+1/rc+1/rs);
  double B=(air.Ta/ra)/tmp;
  double C=(1/rs)/tmp;
  double D=(1/rc)/tmp;
  return (Tg*(1-D)+A-B)/C;    
}
