#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
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
    return 12/pi*0.0820*dr()*(tmp1+tmp2);
  }
  double fcd(void){
    return (1.35*S()/o()-0.35);
  }
  double fcd_beta_lt_03(void){
    int H0 = H;
    H=1;
    while(beta()<0.3){H++;}
    double tmp = (1.35*S()/So()-0.35);
    H = H0;
    return tmp;
  }
  double So(void){
    return (0.75+(2e-5)*ZZ)*S();}
  double Sn(double alb){
    if(SIn<0){return So()*(1.0-alb);}
    else{return SIn*(1.0-alb);}
  }
  double Ln(void){
    if (beta()>=0.3){
      return boltzh*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a()))*fcd();
    }else{
      return boltzh*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a()))*fcd_beta_lt_03();
    }
  }
  double eps_atm(void){return 0.70+0.000595*e_a()*exp(1500/(Ta+Tk));}
  double Rn(void){return Sn()-Ln();}//MJ/m2/h
  double Lsky(void){return eps_atm()*boltz*pow(Ta+Tk,4.0);}
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


class Soil{
  double T();
  double us(double uc, double hc, double LAI, double leaf, double hc){
    a = 0.28*pow(LAI,2.0/3.0)*pow(hc/leaf,1.0/3.0);
    return uc*exp(-a*(1-0.05/hc))};
  double rs(double uc, double hc, double LAI, double leaf, double hc){
    double ap = 0.004;
    double bp = 0.012;
    return 1/(ap+bp*us(uc, hc, LAI, leaf, hc));}
};

class Air{
public:
  Air(double, double, double, double, double);
  ~Air(void);
  double Ta;//C
  double RH;//%
  double uz;//m/s
  double z;//m
  double ZZ;
  double u2(void){if(z==2){return uz;}else{return uz*4.87/log(67.8*z-5.42);}} //m/s, m
  double delta(void){return 4098*(0.6108*exp(17.27*Ta/(Ta+237.7)))/pow(Ta+273.3,2);} //kpa/C
  double P(void){return 101.3*pow(((293-0.0065*ZZ)/293),5.26);}//kpa, 
  double gamma(void){return P()*gammac;}//kpa/c
  double DT(void){return delta()/(delta()+gamma*(1+0.34*u2()));}
  double PT(void){return gamma/(delta()+gamma*(1+0.34*u2()));}
  double TT(void){return (900/(Ta+273))*u2();}
  double e_T(double T){return 0.6108*exp(17.27*T/(T+237.3));}
  double e_s(void){return (e_T(Ta));}
  double e_a(void){return (e_T(Ta)*RH/100)/2;}
  double zm(double hc){return 0.13*hc;}
  double zh(double hc){return zm(hc)/7;}
  double d(double hc){return 0.65*hc;}
  double ra(double hc){return log((z-d(hc))/zm(hc))*log((z-d(hc))/zh(hc))/(vonk*vonk)/rhoa/cp/uz;}//C&N 1998, nutral stability
};

Air::Air(double TTa, double RRH, double uuz, double zz, double ZZZ){
  double Ta = TTa;//C
  double RH = RRH;//%
  double uz = uuz;//m/s
  double z = zz;//m
  double ZZ = zzz;//m
}

Air::~Air(void){
}

class Canopy{
public:
  Canopy(double,double,double,double,double);
  ~Canopy(void);
  double x;//ellipsoidal leaf angle parameter
  double row;//row width m
  double wc;//foliage width ,
  double leaf;//leaf size
  double fc;//crown fraction
  double ff;//total fraction
  double ke;//extinction
  double hc;//foliage height m
  double Kbe(double thetar){return sqrt(x*x+pow(tan(thetar),1))/(x+1.774*pow(x+1.182,-0.733));}//Campbell and Norman 98
  double alb(void){return 0.2;}//albedo from 
  double omega0(void){return (1-porosity())*log(1-ff)/log(porosity())/ff;}//Fuentes 2008
  double omega(double thetar){
    double D = 1;
    double p = 3.80 - 0.46*D;
    double tmp1 = omega0()+(1-omega0())*exp(-2.2*pow(thetar,p))
      return omega0()/tmp1;
  }
  double LAI(void){return -omega0()*fc*log(porosity())/ke;}
  double porosity(void){return ff/fc;}
  double fveg(double thetar){return 1-exp(-Kb(thetar)*omega(thetar)*LAI());}
  double CWSI(double Tcan, double Tdry, double Twet){return (Tcan-Twet)/(Tdry-Twet);}
  double Kc(void){return 0.115+0.235*LAI();}//Ayars paper 2005
  double zm(){return 0.13*hc;}
  double zh(){return zm()/7;}
  double d(){return 0.65*hc;}
  double uc(double uz, double z){return uz*(log((hc-d())/zm()))/(log((z-d())/zm()));}
  double rc(void){
    double C = 90;//sqrt(s)/m grace 1981 via campbell norman 95
    double a = 0.28*pow(LAI,2.0/3.0)*pow(hc/leaf,1.0/3.0);
    double udzm = uc()*exp(-a*(1-(d+zm())/hc));
    return C/LAI()*sqrt(leaf/udzm);}
  double tau_c(void){return Fpar*(Wdir_par*tau_c_dir_par()+Wdiff_par*tau_c_diff_par())+Fnir*(Wdir_nir*tau_c_dir_nir()+Wdiff_nir*tau_c_diff_nir());}
  double tau_c_dir_par(void){return ;}
  double tau_c_diff_par(void){return ;}
  double tau_c_dir_nir(void){return ;}
  double tau_c_diff_nir(void){return ;}
  double rho_c_par_star(void){return ;}
  double rho_hor_par(void){return ;}
  double eta_LAI(double theta_s){return omega(theta_s)/cos(theta_s)*LAI();}
  double alpha_c(void){return Fpar*(Wdir_par*rho_c_dir_par()+Wdiff_par*rho_c_diff_par())+Fnir*(Wdir_nir*rho_c_dir_nir()+Wdiff_nir*rho_c_diff_nir());}
  double rho_c_dir_par(void){return ;}
  double xi_par(double rho_s_par){return ;}
  
};

Canopy::Canopy(double row1, double  wc1, double fc1, double ff1, double hc1){
  row=row1;
  wc=wc1;
  fc=fc1;
  ff=ff1;
  hc=hc1;
  ke=0.7;
  x=2;
}

Canopy::~Canopy(void){
}

double gamma_star(double gamma, double ra, double rc){return gamma*(rc/ra);}//c&n 1998//kpa/c
double Tdry(double d, double z0, double Rnc){return Ta+ra*Rnc/(rhoa*cp);}
double Twet(double rc, double d, double z0, double Rnc, double gamma_star){
  double tmp1 = ra*gamma_star*Rnc/(rhoa*cp*(gamma_star+delta()));
  double tmp2 = (e_s()-e_a())/(gamma_star+delta());
  return Ta + tmp1 - tmp2;
}

double Snc(double LAI){return ;}//????
double Sns(double LAI){return ;}//????
double Lnc(double LAI){return ;}//????
double Lns(double LAI){return ;}//????
double Rnc(double LAI){return ;}//????
double Rns(double LAI){return ;}//????


int main(void){

  Air air(Ta, RH, uz, z, ZZ)
  Rad rad(Y, M, D, H, lat, lonm, lonz, SIn)
  Canopy can(row, wc, fc, ff, hc);
   
  ifstream ifile;
  ofstream ofile;

  
  row=3.5;
  wc=1;
  fc=.5;
  ff=.2;
  hc=1.5;
  air.z=2;//m
  air.ZZ=1066;
  air.lat = 35.15;//deg-latitude
  air.lonm = -102.13;
  air.lonz = -90;//deg-longitude of center of time zone

  air.Kc  = can.Kc();
  air.alb = can.alb();

  ifile.open("../data/072014Bushland.csv",ios_base::in); 
  ofile.open("../data/test.csv",ios_base::out);
  ofile << "Date/Time Rn ET Kc LAI Lsky Twey Tdry Tcan CWSI" << std::endl; 
  string line;
  int dum;
  getline(ifile, line);//header
  while (getline(ifile, line)){
    sscanf(line.c_str(),"%d/%d/%d %d:%d:%d,%lf,%lf,%lf,%lf",&air.M,&air.D,&air.Y,&air.H,&dum,&dum,&air.RsIn,&air.uz,&air.Ta,&air.RH);  
    Ts=air.Ta;
    Tirt=air.Ta;
    ofile << air.M << '/'<< air.D << '/' << air.Y << ' ' << air.H << ','<< air.Rn();
    ofile << ','<<  air.ET() << ','<< can.LAI() << ',' << air.Ta<< ','<< air.Tdry(can.d(),can.z0());
    ofile << ','<< air.Twet(can.rc(), can.d(), can.z0()) << ','<< can.Tcan(Tirt, Ts, air.Lsky()) << ','<<  air.Lsky();
    ofile << ','<< can.CWSI(can.Tcan(Tirt, Ts, air.Lsky()), air.Tdry(can.d(),can.z0()), air.Twet(can.rc(), can.d(), can.z0()));
    ofile << ','<< can.fdhc() << std::endl;
  }
  ofile.close();
  ifile.close();
  return 0;
}
