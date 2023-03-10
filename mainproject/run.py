from waitress import serve
import flaskserver_pi_V5
serve(flaskserver_pi_V5.app, host='0.0.0.0', port=5000)