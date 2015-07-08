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

using namespace std;

string APIKEY = "eea590fdddcc01bb";
string pws = "KOKCYRIL3";
string command = "wget http://api.wunderground.com/api/" + APIKEY + "/conditions/q/pws:" + pws + ".json";
string file = "pws:" + pws + ".json";

int main(void){

  char tmp[1024];
  strcpy(tmp, command.c_str());
  //pull from Wunderground
  int ret = system("rm *.json*");
  ret = system(tmp);
  strcpy(tmp, file.c_str());
  ifstream in(tmp);
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
  char tmp1[24];
  strcpy(tmp1, localtimeStr.c_str());
  
  struct tm tm;
  time_t localtime;

  if(strptime(tmp1,"%a, %b %D %Y %H:%M:%S %z",&tm)==NULL){cout << "oops" << std::endl;}

  localtime = mktime(&tm);
  cout << localtimeStr << localtime << std::endl;
  cout << SIn << std::endl;
  cout << Ta << std::endl;
  cout << Pmb << std::endl;
  cout << uz << std::endl;
  cout << RH << std::endl;
  cout << ZZ << std::endl;
  cout << lonm << std::endl;
  cout << lat << std::endl;
  cout << tm.tm_year << std::endl;
  return 0;
}
