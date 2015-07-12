#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <math.h>
#include <json.h>

using namespace std;

string APIKEY = "eea590fdddcc01bb";
string pws = "KOKCYRIL3";
string command = "wget http://api.wunderground.com/api/" + APIKEY + "/conditions/q/pws:" + pws + ".json";
string file = "pws:" + pws + ".json";

int pull(void){
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
  cout << localtimeStr << localtime << endl;
  cout << SIn << endl;
  cout << Ta << endl;
  cout << Pmb << endl;
  cout << uz << endl;
  cout << RH << endl;
  cout << ZZ << endl;
  cout << lonm << endl;
  cout << lat << endl;
  cout <<  Y << " " << M << " " << D << " " << H << endl;

  double z = 2;
  double hc = 1.8;
  double fc = .5;
  double ff = .3;
  double Tirt = 23;

  Air air(Ta, RH, uz, z, ZZ, hc);
  Rad rad(Y, M, D, H, lat, lonm, lonm, SIn);

  Canopy grapes(fc, ff, hc, Ta);
  Canopy grass(1, .1, .1, Ta);

  /*
  //Soil soil(Ta);
  Rnc = rad.Rnc(grapes.LAI());
  grapes.Tg = Twet(air.ra(), grapes.rc(), grapes.d(), grapes.z0(), Rnc, gamma_star(air.ra(),grapes.rc()));
  double Told = grapes.Tg;
  double Tnew = grapes.Tg*2;
  double delmax = .01;
  int nmax = 100;
  n = 0;
  while(abs(Tnew-Told)>delmax && n<nmax){
    Told = grapes.Tg;
    grass.Tg = Tgrass(grapes, air, rad, grass, eps_rad, Trad, grapes.Tg);
    grapes.Tg = grapes.T(grass.Tg, grass.eps_c, Trad, eps_rad);
    Tnew = grapes.Tg;
    n++;
  }
  */
  return 0;
}
