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
  Wunder(string x, string y):APIKEY(x), pws(y){}
  ~Wunder(void){};
  int pull(void);
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
  string APIKEY;
  string pws;
};

int Wunder::pull(void){
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
  ss >> ZZ;
  ss.clear();
  string lonmStr = wunder["current_observation"]["display_location"]["longitude"].asString();
  ss << lonmStr;
  ss >> lonm;
  ss.clear();
  string latStr =  wunder["current_observation"]["display_location"]["latitude"].asString();
  ss << latStr;
  ss >> lat;
  ss.clear();
  string SInStr = wunder["current_observation"]["solarradiation"].asString();
  ss << SInStr;
  ss >> SIn;
  ss.clear();
  Ta = wunder["current_observation"]["temp_c"].asDouble();
  Ta+=Tk;
  string PmbStr =  wunder["current_observation"]["pressure_mb"].asString();
  ss << PmbStr;
  ss >> Pmb;
  ss.clear();
  uz = wunder["current_observation"]["wind_kph"].asDouble();
  uz = uz/60/60*1000;
  string RHStr = wunder["current_observation"]["relative_humidity"].asString();
  ss << RHStr;
  ss >> RH;
  ss.clear();

  string localtimeStr = wunder["current_observation"]["local_time_rfc822"].asString();
  struct tm tm;
  memset(&tm, 0, sizeof(struct tm));
  time_t localtime;

  if(!strptime(localtimeStr.c_str(),"%a, %0d %b %Y %T",&tm)){cout << "oops, can't parse date" << std::endl;}
  Y =tm.tm_year+1900;
  D =tm.tm_mday;
  M =tm.tm_mon+1;
  H = tm.tm_hour;

  localtime = mktime(&tm);
  cout << SIn << endl;
  cout << Ta << endl;
  cout << Pmb << endl;
  cout << uz << endl;
  cout << RH << endl;
  cout << ZZ << endl;
  cout << lonm << endl;
  cout << lat << endl;}
//asce etsz

class Air{
public:
  double Ta;//K
  double RH;//%
  double uz;//m/s
  double z;//m
  double ZZ;//m
  double hc;//m
  double Pmb;//mb
  Air(Wunder &wun, double z, double hc): Ta(wun.Ta), RH(wun.RH), uz(wun.uz), z(z), hc(hc), Pmb(wun.Pmb), ZZ(wun.ZZ){};
  ~Air(void){};
  double delta(void){return 4098*(0.6108*exp(17.27*(Ta-Tk)/((Ta-Tk)+237.7)))/pow((Ta-Tk)+273.3,2);} //kpa/K
  double P(void){
    if (Pmb < 0){return 101.3*pow(((293-0.0065*ZZ)/293),5.26);}
    else{return Pmb*0.1;}
  }//kpa, 
  double gamma(void){return P()*gammac;}//kpa/K
  double e_T(double T){return 0.6108*exp(17.27*(T-Tk)/((T-Tk)+237.3));}
  double e_s(void){return (e_T(Ta));}
  double e_a(void){return (e_T(Ta)*RH/100)/2;}
  double zm(void){return 0.13*hc;}
  double zh(void){return zm()/7;}
  double d(void){return 0.65*hc;}
  double ra(void){
    double tmp1 = log((z-d())/zm())*log((z-d())/zh());
    return tmp1/(vonk*vonk)/uz;
  }//C&N 1998, nutral stability
  //double eps_atm(void){return 0.70+0.000595*e_a()*exp(1500/(Ta);}
};

class Rad{
public:
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
  double Ta;//K
  double alb;
  double LAI;
  Rad(Wunder &wun,double e_a, double alb, double LAI): D(wun.D), M(wun.M), Y(wun.Y), H(wun.H), lat(wun.lat), lonm(wun.lonm), lonz(wun.lonm), SIn(wun.SIn), ZZ(wun.ZZ), e_a(e_a), Ta(wun.Ta), alb(alb), LAI(LAI){};
  ~Rad(void){};
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
      return boltz*pow(Ta,4)*(0.34-0.14*sqrt(e_a))*fcd();
    }else{
      return boltz*pow(Ta,4)*(0.34-0.14*sqrt(e_a))*fcd_beta_lt_03();
    }
  }
  double Rn(void){return Sn()-Ln();}//W/m2
  double Rnc(void){return Rn()*(1-exp(-0.6*LAI/sqrt(2*cos(Beta()))));}//C&N 99
};

class Canopy{
public:
  double Tg;
  double x;//ellipsoidal leaf angle parameter
  double leaf;//leaf size
  double fc;//crown fraction
  double ff;//total fraction
  double ke;//extinction
  double hc;//foliage height m
  double eps_c;//emissivity
  Canopy(double fc, double ff, double hc, double Tg, double leaf): fc(fc), ff(ff), hc(hc), Tg(Tg), leaf(leaf){ke = 0.5; eps_c = 0.98;};
  ~Canopy(void){};
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
    double G = boltz*eps_rad*pow(Trad,4);
    double F = boltz*eps_s*pow(Ts,4)*(1-fveg());
    double E = boltz*eps_c*fveg();
    return pow((G-F)/E,0.25);}
};

class Soil{
public:
  double Ts;
  double eps_s;
  double LAI;
  double hc;
  double leaf;
  double uc;
  Soil(double Ts, double LAI, double hc, double leaf, double uc):Ts(Ts), LAI(LAI), hc(hc), leaf(leaf), uc(uc){eps_s = 0.98;};
  ~Soil(void){};
  double us(void){
    double a = 0.28*pow(LAI,2/3)*pow(hc/leaf,1/3);
    return uc*exp(-a*(1-0.05/hc));
  }
  double rs(Canopy can, Air air){
    double ap = 0.004;
    double bp = 0.012;
    return 1/(ap+bp*us());}
  double Ls(void){return eps_s*boltz*pow(Ts,4.0);}
};

double gamma_star(double gamma, double ra, double rc){return gamma*(1+rc/ra);}//c&n 1998//kpa/c
double Tdry(Air &air, double Rnc){return air.Ta+air.ra()*Rnc/(rhoa*cp);}
double Twet(Air &air, Canopy &grapes, double Rnc){
  double ra = air.ra();
  double rc = grapes.rc(air.uz, air.z);
  double delta = air.delta();
  double gs = gamma_star(air.gamma(), ra, rc);
  double tmp1 = ra*gs*Rnc/(rhoa*cp*(gs+delta));
  double tmp2 = (air.e_s()-air.e_a())/(gs+delta);
  return air.Ta + tmp1 - tmp2;
}
 
double CWSI(double Tcan, double Tdry, double Twet){return (Tcan-Twet)/(Tdry-Twet);}

double Tsoil(Canopy &grapes, Air &air, Rad &rad, Soil &soil, double eps_rad, double Trad){
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
double Tgrass(Canopy &grapes, Air &air, Rad &rad, Canopy &grass, double eps_rad, double Trad){
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
  double leaf = 0.075;
  double fc = .5;
  double ff = .3;
  double eps_rad = 0.95;
  Wunder cyril(jcAPIKEY, OKpws);
  cyril.pull();
  
  Canopy grapes(fc, ff, hc, cyril.Ta, leaf);
  Canopy grass(1, .1, .1, cyril.Ta, .05);
  
  Air air(cyril, z, hc);
  Rad rad(cyril, air.e_a(), grapes.alb(), grapes.LAI());

  //Soil soil(Ta);
  
  grapes.Tg = Twet(air, grapes, rad.Rnc());
  double Trad = (Tdry(air, rad.Rnc())+grapes.Tg)/2;
  
  double Told = grapes.Tg;
  double Tnew = grapes.Tg*2;
  double delmax = .01;

  int nmax = 10;
  int n = 0;
  
  cout<<grass.Tg << endl;
  cout<<grapes.Tg << endl;
  
  while(abs(Tnew-Told)>delmax && n<nmax){
    Told = grapes.Tg;
    grass.Tg = Tgrass(grapes, air, rad, grass, eps_rad, Trad);
    grapes.Tg = grapes.T(grass.Tg, grass.eps_c, Trad, eps_rad);
    Tnew = grapes.Tg;
    n++;
    cout<<grass.Tg << endl;
    cout<<grapes.Tg << endl;
  }
  
  return 0;
}
