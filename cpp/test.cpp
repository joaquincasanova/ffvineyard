/*#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>*/ 
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <tinyxml2.h>

using namespace std;
using namespace tinyxml2;

string APIKEY = "eea590fdddcc01bb";
string pws = "KOKCYRIL3";
string command = "wget http://api.wunderground.com/api/" + APIKEY + "/conditions/q/pws:" + pws + ".xml";

int main(void){
  char tmp[1024];
  strcpy(tmp, command.c_str());
  //pull from Wunderground
  int ret = system("rm *.xml*");
  ret = system(tmp);

  XMLDocument doc;
  doc.LoadFile( "pws:KOKCYRIL3.xml" );
  dump_to_stdout("pws:KOKCYRIL3.xml");

  return 0;
}
