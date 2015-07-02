#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <vector>
#include <tinyxml2.h>

using namespace std;
using namespace tinyxml2;

int main(void){

  //pull from Wunderground
  int ret = system("rm pws:KOKCYRIL3.xml*");
  ret = system("wget http://api.wunderground.com/api/eea590fdddcc01bb/conditions/q/pws:KOKCYRIL3.xml");

  XMLDocument doc;
  doc.LoadFile( "pws:KOKCYRIL3.xml" );

  
  cout << "Parsing..." << endl;

  
  /*local_time_rfc822
latitude
longitude
elevation
temp_c
relative_humidity
wind_kph
pressure_mb
solarradiation*/

  return 0;
}
