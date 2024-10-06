import os, json, requests, random, time, runpod

import torch
from PIL import Image
import numpy as np
import shutil

import nodes
from nodes import NODE_CLASS_MAPPINGS
from nodes import load_custom_node

import asyncio
import execution
import server
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server)

load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Depth-Pro")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Depthflow-Nodes")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite")

LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()

LoadDepthPro = NODE_CLASS_MAPPINGS["LoadDepthPro"]()
DepthPro = NODE_CLASS_MAPPINGS["DepthPro"]()
MetricDepthToInverse = NODE_CLASS_MAPPINGS["MetricDepthToInverse"]()

Depthflow = NODE_CLASS_MAPPINGS["Depthflow"]()
DepthflowMotionPresetDolly = NODE_CLASS_MAPPINGS["DepthflowMotionPresetDolly"]()
DepthflowMotionPresetZoom = NODE_CLASS_MAPPINGS["DepthflowMotionPresetZoom"]()
DepthflowMotionPresetCircle = NODE_CLASS_MAPPINGS["DepthflowMotionPresetCircle"]()

VHS_VideoCombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

with torch.inference_mode():
    depth_pro_model = LoadDepthPro.load_model("fp16")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image=values['input_image']
    input_image=download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    motion = values['motion']
    intensity = values['intensity']
    reverse = values['reverse']
    smooth = values['smooth']
    loop = values['loop']
    depth = values['depth']
    phase = values['phase']
    phase_x = values['phase_x']
    phase_y = values['phase_y']
    phase_z = values['phase_z']
    amplitude_x = values['amplitude_x']
    amplitude_y = values['amplitude_y']
    amplitude_z = values['amplitude_z']
    static_value = values['static_value']

    strength = 1.0
    feature_threshold = 0.0
    feature_param = "intensity"
    feature_mode = "relative"
    kwargs = {
        'strength': strength,
        'feature_threshold': feature_threshold,
        'feature_param': feature_param,
        'feature_mode': feature_mode
    }

    input_image = LoadImage.load_image(input_image)[0]
    depth_map = DepthPro.estimate_depth(depth_pro_model, input_image)[0]
    depth_map = MetricDepthToInverse.convert_depth(depth_map)[0]

    if motion == "dolly":
        motion = DepthflowMotionPresetDolly.create_internal(intensity, reverse, smooth, loop, depth, **kwargs)[0]
    elif motion == "zoom":
        motion = DepthflowMotionPresetZoom.create_internal(intensity, reverse, smooth, phase, loop, **kwargs)[0]
    elif motion == "circle":
        motion = DepthflowMotionPresetCircle.create_internal(intensity, reverse, smooth, phase_x, phase_y, phase_z, amplitude_x, amplitude_y, amplitude_z, static_value, **kwargs)[0]

    animation_speed = 1.0
    input_fps = 30
    output_fps = 30
    num_frames = 30
    quality = 50
    ssaa = 1.0
    invert = 0.0
    tiling_mode = "mirror"
    depth_flow_image = Depthflow.apply_depthflow(input_image, depth_map, motion, animation_speed, input_fps, output_fps, num_frames, quality, ssaa, invert, tiling_mode, effects=None)[0]
    out_video = VHS_VideoCombine.combine_video(images=depth_flow_image, frame_rate=15, loop_count=0, filename_prefix="DepthFlow", format="video/h264-mp4", save_output=True, prompt=None, unique_id=None)
    source = out_video["result"][0][1][1]
    destination = '/content/ComfyUI/output/depth-flow-tost.mp4'
    shutil.move(source, destination)

    result = '/content/ComfyUI/output/depth-flow-tost.mp4'
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})