#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp> 
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <math.h>
#include <json.h>
//#include "./tseb.h"

using namespace std;

double pi = 3.1459265;
double boltz = 0.00000005670373;// W K-4 m-2
double vonk = 0.4;
double Tk = 273.16;
double gammac = 0.000665;
double cp = 1005; //J/kg/K
double rhoa = 1.205; //kg/m3

string jcAPIKEY = "eea590fdddcc01bb";
string OKpws = "KOKCYRIL3";

class Wunder{
public:
  Wunder(string APIKEY, string pws);
  ~Wunder(void);
  double SIn;
  double Ta;
  double Pmb;
  double uz;
  double RH;
  double ZZ;
  double lonm;
  double lat;
  int Y;
  int M;
  int D;
  int H;
};

Wunder::Wunder(string APIKEY, string pws){
  string command = "wget http://api.wunderground.com/api/" + APIKEY + "/conditions/q/pws:" + pws + ".json";
  string file = "pws:" + pws + ".json";

  //pull from Wunderground
  int ret = system("rm *.json*");
  ret = system(command.c_str());
  ifstream in(file.c_str());
  Json::Value wunder;
  in >> wunder;
  stringstream  ss;

  string ZZStr = wunder["current_observation"]["display_location"]["elevation"].asString();
  ss << ZZStr;
  double ZZ;
  ss >> ZZ;
  ss.clear();
  string lonmStr = wunder["current_observation"]["display_location"]["longitude"].asString();
  ss << lonmStr;
  double lonm;
  ss >> lonm;
  ss.clear();
  string latStr =  wunder["current_observation"]["display_location"]["latitude"].asString();
  ss << latStr;
  double lat;
  ss >> lat;
  ss.clear();
  string SInStr = wunder["current_observation"]["solarradiation"].asString();
  ss << SInStr;
  double SIn;
  ss >> SIn;
  ss.clear();
  double Ta = wunder["current_observation"]["temp_c"].asDouble();
  string PmbStr =  wunder["current_observation"]["pressure_mb"].asString();
  ss << PmbStr;
  double Pmb;
  ss >> Pmb;
  ss.clear();
  double uz = wunder["current_observation"]["wind_kph"].asDouble();
  uz = uz/60/60*1000;
  string RHStr = wunder["current_observation"]["relative_humidity"].asString();
  ss << RHStr;
  double RH;
  ss >> RH;
  ss.clear();

  string localtimeStr = wunder["current_observation"]["local_time_rfc822"].asString();
  struct tm tm;
  memset(&tm, 0, sizeof(struct tm));
  time_t localtime;

  if(!strptime(localtimeStr.c_str(),"%a, %0d %b %Y %T",&tm)){cout << "oops, can't parse date" << std::endl;}
  int Y =tm.tm_year+1900;
  int D =tm.tm_mday;
  int M =tm.tm_mon+1;
  int H = tm.tm_hour;

  localtime = mktime(&tm);
  cout << SIn << endl;
  cout << Ta << endl;
  cout << Pmb << endl;
  cout << uz << endl;
  cout << RH << endl;
  cout << ZZ << endl;
  cout << lonm << endl;
  cout << lat << endl;
  cout <<  Y << " " << M << " " << D << " " << H << endl;
}

Wunder::~Wunder(void){
} 
//asce etsz

class Air{
public:
  Air(Wunder wun, double z, double hc);
  ~Air(void);
  double Ta;//C
  double RH;//%
  double uz;//m/s
  double z;//m
  double ZZ;//m
  double hc;//m
  double Pmb;//mb
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

Air::Air(Wunder wun, double zz, double hhc){
  Ta = wun.Ta;//C
  RH = wun.RH;//%
  uz = wun.uz;//m/s
  z = zz;//m
  hc = hhc;//m
  Pmb = wun.Pmb;//mb
  ZZ = wun.ZZ;//m
}

Air::~Air(void){
}


class Rad{
public:
  Rad(Wunder wun, double ee_a, double aalb, double LLAI);
  ~Rad(void);
  int D;
  int M;
  int Y;
  int H;
  double lat;//deg
  double lonm;//deg-longitude of measurement site
  double lonz;//deg
  double ZZ;//m
  double SIn;//MJ/m2/h
  double e_a;//kpa
  double Ta;//C
  double alb;
  double LAI;
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
  double Beta(void){
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
    while(Beta()<0.3){H++;}
    double tmp = (1.35*S()/So()-0.35);
    H = H0;
    return tmp;
  }
  double So(void){return (0.75+(2e-5)*ZZ)*S();}
  double Sn(void){
    if(SIn<0){return So()*(1.0-alb);}
    else{return SIn*(1.0-alb);}
  }
  double Ln(void){
    if (Beta()>=0.3){
      return boltz*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a))*fcd();
    }else{
      return boltz*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a))*fcd_beta_lt_03();
    }
  }
  double Rn(void){return Sn()-Ln();}//W/m2
  double Rnc(void){return Rn()*(1-exp(-0.6*LAI/sqrt(2*cos(Beta()))));}//C&N 99
};

Rad::Rad(Wunder wun, double ee_a, double aalb, double LLAI){
  D=wun.D;
  M=wun.M;
  Y=wun.Y;
  H=wun.H;
  lat=wun.lat;//deg
  lonm=wun.lonm;//deg-longitude of measurement site
  lonz=wun.lonm;//deg-long of timezone - not right
  SIn=wun.SIn;//MJ/m2/h
  ZZ=wun.ZZ;
  e_a=ee_a;
  Ta=wun.Ta;
  alb=aalb;
  LAI=LLAI;
  cout << D << "/" << M << "/" << Y << endl;
  cout << LAI << endl;
}

Rad::~Rad(void){
}

class Canopy{
public:
  Canopy(double ffc, double fff, double hhc, double lleaf, double TTg);
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
    double tmp1 = omega0()+(1-omega0())*exp(-2.2*pow(theta,p));
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
  double fveg(void){return 1-tau_d();}
  double Kc(void){return 0.115+0.235*LAI();}//Ayars paper 2005
  double zm(void){return 0.13*hc;}
  double zh(void){return zm()/7;}
  double d(void){return 0.65*hc;}
  double uc(double uz, double z){return uz*(log((hc-d())/zm()))/(log((z-d())/zm()));}
  double rc(double uz, double z){
    double C = 90;//sqrt(s)/m grace 1981 via campbell norman 95
    double a = 0.28*pow(LAI(),2.0/3.0)*pow(hc/leaf, 1.0/3.0);
    double udzm = uc(uz,z)*exp(-a*(1-(d()+zm())/hc));
    return C/LAI()*sqrt(leaf/udzm);}
  double T(double Ts, double eps_s, double Trad, double eps_rad){
    double G = boltz*eps_rad*pow(Trad+Tk,4);
    double F = boltz*eps_s*pow(Ts+Tk,4)*(1-fveg());
    double E = boltz*eps_c*fveg();
    return pow((G-F)/E,0.25)-Tk;}
};

Canopy::Canopy(double ffc, double fff, double hhc, double lleaf, double TTg){
  fc=ffc;
  ff=fff;
  hc=hhc;
  ke=0.7;
  x=2;
  eps_c=0.98;
  Tg=TTg;
  leaf=lleaf;
}

Canopy::~Canopy(void){
}

class Soil{
public:
  Soil(double TTs, double LLAI, double hhc, double lleaf, double uuc);
  ~Soil(void);
  double Ts;
  double eps_s;
  double LAI;
  double hc;
  double leaf;
  double uc;
  double us(void){
    double a = 0.28*pow(LAI,2/3)*pow(hc/leaf,1/3);
    return uc*exp(-a*(1-0.05/hc));
  }
  double rs(Canopy can, Air air){
    double ap = 0.004;
    double bp = 0.012;
    return 1/(ap+bp*us());}
  double Ls(void){return eps_s*boltz*pow(Ts+Tk,4.0);}
};

Soil::Soil(double TTs, double LLAI, double hhc, double lleaf, double uuc){
  Ts=TTs;
  LAI=LLAI;
  hc=hhc;
  leaf=lleaf;
  uc=uuc;
  eps_s=0.98;
}

Soil::~Soil(void){
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
  double A=-rc*gs*rad.Rnc()/(delta+gs)/rhoa/cp+rc/ra*(air.e_s()-air.e_a())/(delta+gs);
  double tmp = (1/ra+1/rc+1/rs);
  double B=(air.Ta/ra)/tmp;
  double C=(1/rs)/tmp;
  double D=(1/rc)/tmp;
  return (grapes.T(soil.Ts, soil.eps_s, Trad, eps_rad)*(1-D)+A-B)/C;    
}
double Tgrass(Canopy grapes, Air air, Rad rad, Canopy grass, double eps_rad, double Trad){
  double ra = air.ra();
  double rc = grapes.rc(air.uz, air.z);
  double rs = grass.rc(air.uz, air.z);
  double LAI = grapes.LAI();
  double delta = air.delta();
  double gs = gamma_star(air.gamma(), ra, rc);
  double A=-rc*gs*rad.Rnc()/(delta+gs)/rhoa/cp+rc/ra*(air.e_s()-air.e_a())/(delta+gs);
  double tmp = (1/ra+1/rc+1/rs);
  double B=(air.Ta/ra)/tmp;
  double C=(1/rs)/tmp;
  double D=(1/rc)/tmp;
  return (grapes.Tg*(1-D)+A-B)/C;    
}

int main(void){

  double z = 2;
  double hc = 1.8;
  double leaf = 7.5;
  double fc = .5;
  double ff = .3;
  double Trad = 23;
  double eps_rad = 0.98;
  Wunder cyril(jcAPIKEY, OKpws);
  Canopy grapes(fc, ff, hc, cyril.Ta, leaf);
  Canopy grass(1, .1, .1, cyril.Ta, .05);

  
  Air air(cyril, z, hc);
  Rad rad(cyril, air.e_a(), grapes.alb(), grapes.LAI());

  //Soil soil(Ta);
  cout << rad.Beta() << endl;
  cout << grapes.LAI() << endl;
  cout << rad.Rnc() << endl;
  
  grapes.Tg = Twet(air, grapes, rad.Rnc());
  cout << grapes.Tg << endl;
  cout << Tdry(air, rad.Rnc()) << endl;
  /*
  double Told = grapes.Tg;
  double Tnew = grapes.Tg*2;
  double delmax = .01;

  int nmax = 100;
  int n = 0;

  while(abs(Tnew-Told)>delmax && n<nmax){
    Told = grapes.Tg;
    grass.Tg = Tgrass(grapes, air, rad, grass, eps_rad, Trad);
    grapes.Tg = grapes.T(grass.Tg, grass.eps_c, Trad, eps_rad);
    Tnew = grapes.Tg;
    n++;
  }
  */
  return 0;
}
