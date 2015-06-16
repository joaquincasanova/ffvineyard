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

double PI = 3.1459265;
double boltz = 0.00000005670373;// W K-4 m-2
double boltzh = 0.0000000002042;// MJ K-4 m-2 h-1
double vonk = 0.4;
double Tk = 273.16;
double psych = 0.000665;
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
  double RsIn;//MJ/m2/h
  int J(void){return D-32+(int)(275*M/9)+2*(int)(3/(M+1))+(int)(M/100-(Y%4)/4+0.975);}
  double Sc(void){
    double b=2*PI*(J()-81)/364;
    return 0.1645*sin(2*b)-0.1255*cos(b)-0.025*sin(b);
  }//hour
  double dr(void){return 1+0.033*cos(2*PI/365*J());}
  double dec(void){return 0.409*sin(2*PI/365*J()-1.39);}
  double phi(void){return PI/180*lat;}
  double omega(void){return PI/12*(H+0.06667*(lonz-lonm)+Sc()-12);}
  double omega_s(void){return acos(-tan(phi())*tan(dec()));}
  double omega_1(void){
    double tmp=omega()-PI*H/24; 
    if(tmp<-omega_s()){
      tmp=-omega_s();
    }
    if(tmp>omega_s()){
      tmp=omega_s();
    }
    return tmp;
  }
  double omega_2(void){
    double tmp=omega()+PI*H/24;
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
  double Rs(void){
    double tmp1 = (omega_2()-min(omega_1(),omega_2()))*sin(phi())*sin(dec());
    double tmp2 = (sin(omega_2()-min(omega_1(),omega_2())))*cos(phi())*cos(dec());
    return 12/PI*0.0820*dr()*(tmp1+tmp2);
  }
  double fcd(void){
    return (1.35*Rs()/Rso()-0.35);
  }
  double fcd_beta_lt_03(void){
    int H0 = H;
    H=1;
    while(beta()<0.3){H++;}
    double tmp = (1.35*Rs()/Rso()-0.35);
    H = H0;
    return tmp;
  }
  double Rso(void){
    return (0.75+(2e-5)*ZZ)*Rs();}
  double Rns(double alb){
    if(RsIn<0){return Rso()*(1.0-alb);}
    else{return RsIn*(1.0-alb);}
  }
  double Rnl(void){
    if (beta()>=0.3){
      return boltzh*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a()))*fcd();
    }else{
      return boltzh*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a()))*fcd_beta_lt_03();
    }
  }
  double eps_atm(void){return 0.70+0.000595*e_a()*exp(1500/(Ta+Tk));}
  double Rn(void){return Rns()-Rnl();}//MJ/m2/h
  double Lsky(void){return eps_atm()*boltz*pow(Ta+Tk,4.0);}
  
};

Rad::Rad(int YY, int MM, int DD, int HH, double llat, double llonm, double llonz, double RRsIn){
  int D=DD;
  int M=MM;
  int Y=YY;
  int H=HH;
  double lat=llat;//deg
  double lonm=llonm;//deg-longitude of measurement site
  double lonz=llonz;//deg
  double RsIn=RRsIn;//MJ/m2/h
}

Rad::~Rad(void){
}


class Soil{
  double Ts();
  double us();
  double rs();
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
  double Tdry(double d, double z0, double Rn){return Ta+ra(d, z0)*Rn/(rhoa*cp);}
  double Twet(double rc, double d, double z0, double Rn){
    double tmp1 = rc*ra(d,z0)*gamma()*Rn/(rhoa*cp*(gamma()*rc+delta()*ra(d,z0)));
    double tmp2 = ra(d,z0)*(e_s()-e_a())/(gamma()*rc+delta()*ra(d,z0));
    return Ta + tmp1 - tmp2;
  }

  //private:
  double u2(void){if(z==2){return uz;}else{return uz*4.87/log(67.8*z-5.42);}} //m/s, m
  double delta(void){return 4098*(0.6108*exp(17.27*Ta/(Ta+237.7)))/pow(Ta+273.3,2);} //kpa/C
  double P(void){return 101.3*pow(((293-0.0065*ZZ)/293),5.26);}//kpa, 
  double gamma(void){return psych*P();}
  double DT(void){return delta()/(delta()+gamma()*(1+0.34*u2()));}
  double PT(void){return gamma()/(delta()+gamma()*(1+0.34*u2()));}
  double TT(void){return (900/(Ta+273))*u2();}
  double e_T(double T){return 0.6108*exp(17.27*T/(T+237.3));}
  double e_s(void){return (e_T(Ta));}
  double e_a(void){return (e_T(Ta)*RH/100)/2;}
  double ra(double d, double z0){return 4.72*pow((log((z-d)/z0)/vonk),2)/(1+0.54*uz);}
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
  double z0(void){return 0.13*hc;}
  double d(void){return 0.63*hc;}
  double uc(void){;}
  double rc(void){;}//s/m Texeira 2007///WRONG use campbell norman
  double Tcan(double Tirt, double Ts, double Lsky){
    double eps_c = 0.98;
    double eps_s = 0.98;
    double eps_i = 0.95;
    double A = 1-fveg();
    double B = (1.0-eps_s)*(1.0-fveg())+fveg()*(1.0-eps_c);
    double Ls = eps_s*boltz*pow(Ts+Tk,4.0);
    double Lirt = eps_i*boltz*pow(Tirt+Tk,4.0);
    double tmp=(Lirt-Lsky*B-Ls*A)/eps_c/fveg()/boltz;
    return pow(tmp,0.25)-Tk;
  }
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
  

int main(void){

  Air air(Ta, RH, uz, z, ZZ)
  Rad rad(Y, M, D, H, lat, lonm, lonz, RsIn)
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
