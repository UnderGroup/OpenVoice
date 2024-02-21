import os
from flask import Flask, jsonify, request, send_from_directory
import argparse
import logging
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from logging.handlers import RotatingFileHandler
from coloredlogs import ColoredFormatter


app = Flask(__name__)
port = int(os.environ.get('FLASK_PORT', 5000))
logger = logging.getLogger('gen')


def setup_logging(module: str, level: str, console: int = 1, log_dir="./log"):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = logging.getLogger(module)
    logger.setLevel(level)
    handler = RotatingFileHandler(f'{log_dir}/.{module}.log', maxBytes=20 * 1024 * 1024, backupCount=10)
    fmt_str = '[%(asctime)s] %(levelname)s [%(name)s.%(module)s.%(funcName)s:%(lineno)d] %(message)s'
    date_fmt_str = '%Y-%m-%dT%H:%M:%S'
    formatter = logging.Formatter(fmt_str, datefmt=date_fmt_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if console == 1:
        formatter = ColoredFormatter(fmt_str, date_fmt_str)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)


@app.route('/gen-voice', methods=['GET'])
def gen_voice():
    content = request.args.get("content")
    speed = float(request.args.get("speed", "1.0"))
    voice_type = int(request.args.get("voice_type", "0"))
    output_path = text_to_voice(content, speed, voice_type)
    output_dir = os.path.dirname(output_path)
    output_file = os.path.basename(output_path)
    return send_from_directory(output_dir, output_file)


def text_to_voice(content: str, speed: float, voice_type: int):
    logger.debug("content=%s speed=%s voice_type=%d", content, speed, voice_type)
    home_path = "./"
    ckpt_base = f'{home_path}/checkpoints/base_speakers/EN'
    ckpt_converter = f'{home_path}/checkpoints/converter'
    file_audio = "demo_speaker0.mp3"
    if voice_type == 1:
        file_audio = "demo_speaker1.mp3"
    elif voice_type == 2:
        file_audio = "demo_speaker2.mp3"
    reference_speaker = f'{home_path}/resources/{file_audio}'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'outputs'

    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)

    source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed',
                                                vad=True)

    save_path = f'{output_dir}/output_en_default.wav'

    # Run the base speaker tts
    src_path = f'{output_dir}/tmp.wav'
    base_speaker_tts.tts(content, src_path, speaker='default', language='English', speed=speed)

    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message)
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text to voice')
    parser.add_argument('--console', type=int, default=1)
    parser.add_argument('--level', type=str, default='DEBUG')
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()
    setup_logging('werkzeug', args.level, args.console)
    setup_logging('gen', args.level, args.console)
    app.run(debug=args.debug, host='0.0.0.0', port=port)
