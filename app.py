from flask import Flask, request, make_response
import os, time, subprocess, random, datetime, glob


app = Flask(__name__)

PROJ_DIR = '/home/dhaba/art-fart/art-DCGAN/'
# PROJ_DIR = '/Users/davishaba/PycharmProjects/demo/'
CP_DIR = PROJ_DIR + 'checkpoints/'
GEN_DIR = PROJ_DIR

NOISE_MODES = ['random', 'line', 'linefull1d', 'linefull']


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/generate')
def generate():
    batch_size = request.args['batchSize']
    img_size = request.args['imSize']
    model_name = request.args['model']
    noise_mode = request.args['noiseMode']

    img_path = gen_img(batch_size, img_size, model_name, noise_mode)
    print("got img path {}".format(img_path))

    resp = make_response(open(img_path).read())
    resp.content_type = "image/jpeg"
    return resp

    # script = 'sleep 5'
    # print(time.time())
    # print("request args: {}".format(request.args))
    # print('running script:\n{}'.format(script))
    # os.system(script)
    # print('done running scripts')
    # print(time.time())
    # return str(random.random())

@app.route("/imgs/<path>")
def images(path):
    print("path is {}".format(path))
    return 'hi'
    # generate_img(path)
    # fullpath = "./imgs/" + path
    # resp = flask.make_response(open(fullpath).read())
    # resp.content_type = "image/jpeg"
    # return resp


def gen_img(batch_size: int, img_size: int, model_name: str, noise_mode: str) -> str:
    assert noise_mode in NOISE_MODES
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cp_num, latest_cp = get_latest_checkpoint_num(model_name)
    f_name = "{}_{}_{}".format(model_name, cp_num, date_str)

    # TODO extra validation for noise modes that only do batch size of 1???

    cmd = "gpu=1 batchSize={} net={} imsize={} name={} display=0 noisemode={}".format(batch_size, latest_cp, img_size, f_name, noise_mode)

    print("Executing cmd:\n{}".format(cmd))
    os.system(cmd)
    print("Done executing cmd")

    return f_name


def get_latest_checkpoint_num(model_name: str):
    prefix = "{}{}_".format(CP_DIR, model_name)
    suffix = "_net_G.t7"
    pattern = "{}*{}".format(prefix, suffix)
    all_checkpoints = glob.glob(pattern)
    assert len(all_checkpoints) > 0, "No checkpoints found matching glob {}".format(pattern)

    cp_nums = []
    for f_name in all_checkpoints:
        cp_num = int(f_name[len(prefix):-len(suffix)])
        print("f name {} has cp_num {}".format(f_name, cp_num))
        cp_nums.append(cp_num)

    assert len(cp_nums) > 0, "this is bad u done fucked up davis"
    latest_cp = "{}{}{}".format(prefix, max(cp_nums), suffix)
    assert os.path.exists(latest_cp), "Expected checkpoint at path {}, but not found".format(latest_cp)

    print("latest check {}".format(latest_cp))
    return max(cp_nums), latest_cp





if __name__ == '__main__':
    get_latest_checkpoint_num('airmax')
    app.run(host='0.0.0.0', port='8004')
