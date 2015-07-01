import urllib2
import json
f = urllib2.urlopen('http://api.wunderground.com/api/eea590fdddcc01bb/conditions/q/pws:KOKCYRIL3.json')
json_string = f.read()
parsed_json = json.loads(json_string)
#location = parsed_json['observation_location']['city']
temp_f = parsed_json['current_observation']['temp_f']
print "Current temperature in is: %s" % (temp_f)
f.close()
