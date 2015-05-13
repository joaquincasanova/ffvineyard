#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>

#define PI 3.1459265
#define boltz 5.670373e−8// W K-4 m-2
#define boltzd 4.903e-9// MJ K-4 m-2 day-1
#define boltzh 2.042e-10// MJ K-4 m-2 h-1
#define vonk 0.4
#define Tk 273.16
#define psych 0.000665
#define cp 1005 //J/kg/K
#define rhoa 1.205 //kg/m3
 
using namespace std;

//asce etsz

class Hourly{
public:
  Hourly(void);
  ~Hourly(void);
  int D;
  int M;
  int Y;
  int H;
  double lat;//deg
  double lonm;//deg
  double lonz;//deg
  double alb;
  double Ta;//C
  double RH;//%
  double uz;//m/s
  double z;//m
  double ZZ;//m
  double Kc;
  double Rnin;//MJ/m2/h
  double ET(void){return Kc*ET_o();}//mm

  double Rn(void){
    if(Rnin<0){
      return Rns()-Rnl();
      }else{
      return Rnin;
    };
  }

  double Lsky(double emiss_atm, double Ta){return;}
  double ra(double d, double z0){return 4.72*(log((z-d)/z0)/vonk)^2/(1+0.54*uz)};
  double Tdry(void){return Ta+ra(double uz, double z)*Rn/(rhoa*cp);}
  double Twet(double rc){return ;}

private:
  double u2(void){return uz*4.87/log(67.8*z-5.42);} //m/s, m
  double delta(void){return 4098*(0.6108*exp(17.27*Tmean()/(Tmean()+237.7)))/pow(Tmean()+273.3,2);} //kpa/C
  double P(void){return 101.3*pow(((293-0.0065*ZZ)/293),5.26);}//kpa, 
  double gamma(void){return psych*P();}
  double DT(void){return delta()/(delta()+gamma()*(1+0.34*u2()));}
  double PT(void){return gamma()/(delta()+gamma()*(1+0.34*u2()));}
  double TT(void){return (900/(a+273))*u2();}
  double e_T(double T){return 0.6108*exp(17.27*T/(T+237.3));}
  double e_s(void){return (e_T(Ta));}
  double e_a(void){return (e_T(Ta)*RH/100)/2;}
  double ET_rad(void){return DT()*Rng();}
  double ET_wind(void){return PT()*TT()*(e_s()-e_a());}
  double ET_o(void){return ET_wind()+ET_rad();}
  double Rng(void){return 0.408*Rn;}
  int J(void){return D-32+(int)(275*M/9)+2*(int)(3/(M+1))+(int)(M/100-(Y%4)/4+0.975);}
  double Sc(void){
    b=2*PI*(J()-81)/364;
    return 0.1645*sin(2*b)-0.1255*cos(b)-0.025*sin(b);
  }//hour
  double dr(void){return 1+0.033*cos(2*PI/365*J());}
  double dec(void){return 0.409*sin(2*PI/365*J()-1.39);}
  double phi(void){return PI/180*lat;}
  double omega(void){return PI/12*(H+0.06667*(lonz-lonm)+Sc)-12;}
  double omega_s(void){return acos(-tan(phi())*tan(dec()));}
  double omega_1(void){
    tmp=omega()-PI*H/24; 
    if(tmp<-omega_s()){
      tmp=-omega_s();
    }
    if(tmp>omega_s()){
      tmp=omega_s();
    }
    return tmp;
  }
  double omega_2(void){
    tmp=omega()+PI*H/24
    if(tmp<-omega_s()){
      tmp=-omega_s();
    }
    if(tmp>omega_s()){
      tmp=omega_s();
    }
    return tmp;
  }
  double beta(void){
    return ;
  }
  double Rs(void){
    return 12/PI*0.0820*dr()*((omega_2()-min(omega_1(),omega_2()))*sin(phi())*sin(dec())+(sin(omega_2()-min(omega_1(),omega_2())))*cos(phi())*cos(dec()));
  }
  double fcd(void){
    return (1.35*Rs()/Rso()-0.35);
  }
  double Rso(void){return (0.75+(2e-5)*ZZ)*Rs();}
  double Rns(void){return Rso()*(1.0-alb);}
  double Rnl(void){return boltzh*pow(Ta+Tk,4)*(0.34-0.14*sqrt(e_a()))*fcd();}
  double Rng(void){return 0.408*Rn();}
};

Hourly::Hourly(void){
}

Hourly::~Hourly(void){
}

class Daily{
public:
  Daily(void);
  ~Daily(void);

  int D;
  int M;
  int Y;
  double Tmax;//C
  double Tmin;//C
  double RHmax;//%
  double RHmin;//%
  double uz;//m/s
  double z;//m
  double ZZ;//m
  double lat;//deg
  double alb;
  double Kc;
  double Rnin;//MJ/m2/day

  double Rn(void){
    if(Rnin<0){
      return Rns()-Rnl();
      }else{
      return Rnin;
    };
  }

  double ET(void){return Kc*ET_o();}//mm

private:
  double Tmean(void){return (Tmin+Tmax)/2;} //degree C
  double u2(void){return uz*4.87/log(67.8*z-5.42);} //m/s, m
  double delta(void){return 4098*(0.6108*exp(17.27*Tmean()/(Tmean()+237.7)))/pow(Tmean()+273.3,2);} //kpa/C
  double P(void){return 101.3*pow(((293-0.0065*ZZ)/293),5.26);}//kpa, 
  double gamma(void){return psych*P();}
  double DT(void){return delta()/(delta()+gamma()*(1+0.34*u2()));}
  double PT(void){return gamma()/(delta()+gamma()*(1+0.34*u2()));}
  double TT(void){return (900/(Tmean()+273))*u2();}
  double e_T(double T){return 0.6108*exp(17.27*T/(T+237.3));}
  double e_s(void){return (e_T(Tmin)+e_T(Tmax))/2;}
  double e_a(void){return (e_T(Tmin)*RHmax/100+e_T(Tmax)*RHmin/100)/2;}
  double ET_rad(void){return DT()*Rng();}
  double ET_wind(void){return PT()*TT()*(e_s()-e_a());}
  double ET_o(void){return ET_wind()+ET_rad();}

  int J(void){return D-32+(int)(275*M/9)+2*(int)(3/(M+1))+(int)(M/100-(Y%4)/4+0.975);}
  double dr(void){return 1+0.033*cos(2*PI/365*J());}
  double dec(void){return 0.409*sin(2*PI/365*J()-1.39);}
  double phi(void){return PI/180*lat;}
  double omega_s(void){return acos(-tan(phi())*tan(dec()));}
  double fcd(void){return (1.35*Rs()/Rso()-0.35);}
  double Rs(void){return 24*60/PI*0.0820*dr()*(omega_s()*sin(phi())*sin(dec())+sin(omega_s())*cos(phi())*cos(dec()));}
  double Rso(void){return (0.75+(2e-5)*ZZ)*Rs();}
  double Rns(void){return Rso()*(1.0-alb);}
  double Rnl(void){return boltzd*(pow(Tmin+Tk,4)+pow(Tmax+Tk,4))/2*(0.34-0.14*sqrt(e_a()))*fcd();}
  double Rng(void){return 0.408*Rn();}

};

Daily::Daily(void){
}

Daily::~Daily(void){
}

class Canopy{
public:
  double row;
  double wc;
  double fc;
  double ff;
  double ke=0.7;
  double klw=0.95;
  double hc;
  double omega0(void){(return1-porosity)*log(1-ff)/log(porosity)/ff;}//Fuentes 2008
  double LAI(void){return -omega0()*fc*log(porosity())/ke;}
  double porosity(void){return ff/fc;}
  double fdhc(void){return ff};
  double CWSI(double Tcan, double Tdry, double Twet){return (Tcan-Twet)/(Tdry-Twet);}
  double Kc(void){return 0.115+0.235*LAI();}//Ayars paper 2005
  double z0(void){return 0.13*hc;}
  double h(void){return 0.63*hc;}
  double rc(void){return 120.0;}//s/m Texeira 2007
  double thetalw(void){return exp(-klw*row/wc*LAI());}
  double Tcan(double Tirt, double Ts){return;}
};
  

int main(void){

  Daily Day;

  cout << "D ";
  cin >> Day.D;
  cout << "M ";
  cin >> Day.M;
  cout << "Y ";
  cin >> Day.Y;
  cout << "Tmax ";
  cin >> Day.Tmax;//C
  cout << "Tmin ";
  cin >> Day.Tmin;//C
  cout << "RHmax ";
  cin >> Day.RHmax;//%
  cout << "RHmin ";
  cin >> Day.RHmin;//%
  cout << "uz ";
  cin >> Day.uz;//m/s
  cout << "z ";
  cin >> Day.z;//m
  cout << "ZZ ";
  cin >> Day.ZZ;//m
  cout << "lat ";
  cin >> Day.lat;//deg
  cout << "alb ";
  cin >> Day.alb;
  cout << "Kc ";
  cin >> Day.Kc;
  cout << "Rnin ";
  cin >> Day.Rnin;//
  cout << Day.Rn() << std::endl;
  cout << "ET " << std::endl; 
  cout << Day.ET() << std::endl;

  return 0;
}
