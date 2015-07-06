import urllib2
import json
APIKEY = 'eea590fdddcc01bb'
pws = 'KOKCYRIL3'
command = 'http://api.wunderground.com/api/' + APIKEY + '/conditions/q/pws:' + pws + '.json'
f = urllib2.urlopen(command)
json_string = f.read()
parsed_json = json.loads(json_string)
temp_c = parsed_json['current_observation']['temp_c']
solarradiation = parsed_json['current_observation']['solarradiation']
pressure_mb = parsed_json['current_observation']['pressure_mb']
wind_kph = parsed_json['current_observation']['wind_kph']
relative_humidity = parsed_json['current_observation']['relative_humidity']
longitude = parsed_json['current_observation']['display_location']['latitude']
latitude = parsed_json['current_observation']['display_location']['longitude']
elevation = parsed_json['current_observation']['display_location']['elevation']

print "Current temperature in %s, %s, %s is: %s" % (latitude, longitude, elevation, temp_c)
print "Current solar radiation in %s, %s, %s is: %s" % (latitude, longitude, elevation, solarradiation)
print "Current relative humidity in %s, %s, %s is: %s" % (latitude, longitude, elevation, relative_humidity)
print "Current wind speed in %s, %s, %s is: %s" % (latitude, longitude, elevation, wind_kph)
print "Current pressure in %s, %s, %s is: %s" % (latitude, longitude, elevation, pressure_mb)

f.close()
