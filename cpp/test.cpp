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

  cout << wunder["current_observation"]["solarradiation"] << std::endl;
  cout << wunder["current_observation"]["temp_c"] << std::endl;
  cout << wunder["current_observation"]["pressure_mb"] << std::endl;
  cout << wunder["current_observation"]["wind_kph"] << std::endl;
  cout << wunder["current_observation"]["relative_humidity"] << std::endl;
  cout << wunder["current_observation"]["display_location"]["elevation"] << std::endl;
  cout << wunder["current_observation"]["display_location"]["longitude"] << std::endl;
  cout << wunder["current_observation"]["display_location"]["latitude"] << std::endl;
  cout << wunder["current_observation"]["local_time_rfc822"] << std::endl;
  return 0;
}
