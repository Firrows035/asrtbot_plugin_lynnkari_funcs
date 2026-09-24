from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
import astrbot.api.message_components as Comp
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.event import MessageChain

import websocket
import uuid
import json
import urllib.request
import urllib.parse
import os
import io
from PIL import Image


server_address = "192.168.1.14:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt, prompt_id):
    p = {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    urllib.request.urlopen(req).read()

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = str(uuid.uuid4())
    queue_prompt(prompt, prompt_id)
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

prompt_text = """
{
  "1": {
    "inputs": {
      "unet_name": "Krea2_Turbo_convrot_int8mixed.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "UNet加载器"
    }
  },
  "2": {
    "inputs": {
      "text": "一个银发女孩的全身照。\n",
      "clip": [
        "3",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP文本编码"
    }
  },
  "3": {
    "inputs": {
      "clip_name": "qwen3vl_4b_uncensored_int8_convrot.safetensors",
      "type": "krea2",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {
      "title": "加载CLIP"
    }
  },
  "4": {
    "inputs": {
      "vae_name": "Wan2_1_VAE_bf16.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "加载VAE"
    }
  },
  "5": {
    "inputs": {
      "conditioning": [
        "2",
        0
      ]
    },
    "class_type": "ConditioningZeroOut",
    "_meta": {
      "title": "条件零化"
    }
  },
  "6": {
    "inputs": {
      "seed": 185234953315678,
      "steps": 8,
      "cfg": 1,
      "sampler_name": "euler_ancestral",
      "scheduler": "beta",
      "denoise": 1,
      "model": [
        "1",
        0
      ],
      "positive": [
        "2",
        0
      ],
      "negative": [
        "5",
        0
      ],
      "latent_image": [
        "9",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "K采样器"
    }
  },
  "7": {
    "inputs": {
      "images": [
        "8",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "预览图像"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "6",
        0
      ],
      "vae": [
        "4",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE解码"
    }
  },
  "9": {
    "inputs": {
      "width": [
        "10",
        0
      ],
      "height": [
        "10",
        1
      ],
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "空Latent图像"
    }
  },
  "10": {
    "inputs": {
      "aspect_ratio": "3:4 (Portrait Standard)",
      "megapixels": 1,
      "multiple": 64
    },
    "class_type": "ResolutionSelector",
    "_meta": {
      "title": "分辨率选择器"
    }
  }
}
"""
async def generate(user_prompt,server_address,client_id):
    prompt = json.loads(prompt_text, strict=False)
    #set the text prompt for our positive CLIPTextEncode
    prompt["2"]["inputs"]["text"] = user_prompt

    #set the seed for our KSampler node
    prompt["6"]["inputs"]["seed"] = 114514

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt)
    ws.close() # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts
    #Commented out code to display the output images:

    # for node_id in images:
    #     for image_data in images[node_id]:
    #         from PIL import Image
    #         import io
    #         image = Image.open(io.BytesIO(image_data))
    #         image.show()
    return images

async def save(images):
    save_dir = r"D:/ComfyUI_AstrBot_Temp"
    os.makedirs(save_dir, exist_ok=True)

    for node_id in images:
        for i, image_data in enumerate(images[node_id]):
            image = Image.open(io.BytesIO(image_data))
            filename = f"node{node_id}_{i}.png"
            path = os.path.join(save_dir, filename)
            image.save(path)
            print("saved:", path)
            return f"D:/ComfyUI_AstrBot_Temp/{filename}"
@register("Ferrin's Toolkit", "Fylavvor", "神秘妙妙工具", "0.0.2")
class MyPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""

    # 注册指令的装饰器。指令名为 helloworld。注册成功后，发送 `/helloworld` 就会触发这个指令，并回复 `你好, {user_name}!`
    # @filter.command("helloworld")
    # async def helloworld(self, event: AstrMessageEvent):
    #     """这是一个 hello world 指令""" # 这是 handler 的描述，将会被解析方便用户了解插件内容。建议填写。
    #     user_name = event.get_sender_name()
    #     message_str = event.message_str # 用户发的纯文本消息字符串
    #     message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
    #     logger.info(message_chain)
    #     yield event.plain_result(f"Hello, {user_name}, 你发了 {message_str}!") # 发送一条纯文本消息

    @filter.command_group("ff", alias={"f","fff"})
    def ff():
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @ff.command("testChain")
    async def testChain(self, event: AstrMessageEvent):
        """测试功能用"""
        chain = [
            Comp.At(qq=event.get_sender_id()),  # At 消息发送者
            Comp.Plain("欢迎使用A.I.S.I.S.！"),
            Comp.Image.fromFileSystem("./data/plugins/asrtbot_plugin_lynnkari_funcs/logo.png"),  # 从本地文件目录发送图片
        ]
        yield event.chain_result(chain)

    @ff.command("add")
    async def add(self, event: AstrMessageEvent, a: int, b: int):
        """简单加法"""
        # /ff add 1 2 -> 结果是: 3
        yield event.plain_result(f"结果是: {a + b}")

    @ff.command("picture", alias={"生成图片"})
    async def picture(self, event: AstrMessageEvent, user_prompt: str):
        """调用本地ComfyUI生成图片"""
        umo = event.unified_msg_origin
        message_chain = MessageChain().message("收到指令。尝试连接中...")
        await self.context.send_message(umo, message_chain)
        image=await generate(user_prompt, server_address, client_id)
        message_chain = MessageChain().message("图片已生成！")
        path=await save(image)
        await self.context.send_message(umo, message_chain)
        message_chain = MessageChain().file_image(f"{path}")
        await self.context.send_message(umo, message_chain)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @ff.command("setip")
    async def setip(self,event:AstrMessageEvent, ip4: int):
        """修改ComfyUI所在ip"""
        global server_address
        server_address = f"192.168.1.{ip4}:8188"
        yield event.plain_result(f"ComfyUI服务器地址已修改：{server_address}")


    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
