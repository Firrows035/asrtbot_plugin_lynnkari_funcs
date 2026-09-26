from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
import astrbot.api.message_components as Comp
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.event import MessageChain
import asyncio
import random
import websocket
import uuid
import json
import urllib.request
import urllib.parse
import os
import io
import aiohttp
from PIL import Image

comfyui_queue=False
server_address = "192.168.1.14:8188"
client_id = str(uuid.uuid4())

function_picture=True


def queue_prompt(prompt, prompt_id):
    """向 ComfyUI 提交一次生成任务。

    这是阻塞调用，必须在工作线程中执行（见 wait_until_executed）。

    Args:
        prompt: ComfyUI 工作流 prompt。
        prompt_id: 本次任务的 id。
    """
    p = {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    urllib.request.urlopen(req).read()

async def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    url = "http://{}/view?{}".format(server_address, url_values)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def get_history(prompt_id):
    """获取某次任务的执行历史（阻塞调用，请在工作线程中执行）。"""
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def wait_until_executed(ws, prompt, prompt_id):
    """提交任务并阻塞等待 ComfyUI 执行结束。

    该函数是同步阻塞的（websocket 收包 + HTTP 请求），必须在工作线程中执行，
    否则会卡住 AstrBot 的事件循环，导致生成期间其它消息无法被处理。

    Args:
        ws: 已经连接好的 ComfyUI websocket 客户端。
        prompt: ComfyUI 工作流 prompt。
        prompt_id: 本次任务的 id。

    Returns:
        ComfyUI 执行历史中本次任务对应的结果。
    """
    queue_prompt(prompt, prompt_id)
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

    return get_history(prompt_id)[prompt_id]

async def get_images(ws, prompt):
    prompt_id = str(uuid.uuid4())
    history = await asyncio.to_thread(wait_until_executed, ws, prompt, prompt_id)
    output_images = {}
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data =await get_image(image['filename'], image['subfolder'], image['type'])
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
      "text": "一个银发女孩的全身照。",
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
      "seed": 201337714489622,
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
      "multiple": 16
    },
    "class_type": "ResolutionSelector",
    "_meta": {
      "title": "分辨率选择器"
    }
  }
}
"""
aspect_ratio_options=["1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)", "3:4 (Portrait Standard)", "4:3 (Standard)", "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)"]
async def generate(user_prompt,server_address,client_id,self,umo):
    prompt = json.loads(prompt_text, strict=False)
    #set the text prompt for our positive CLIPTextEncode
    prompt["20"]["inputs"]["text"] = user_prompt

    #set the seed for our KSampler node
    prompt["6"]["inputs"]["seed"] = random.randint(1,2**32-1)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    await self.context.send_message(umo, MessageChain().message("服务器连接成功，准备生成..."))
    images = get_images(ws, prompt)
    ws.close() # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts
    #Commented out code to display the output images:

    # for node_id in images:
    #     for image_data in images[node_id]:
    #         from PIL import Image
    #         import io
    #         image = Image.open(io.BytesIO(image_data))
    #         image.show()
    yield images

def save(images):
    """把生成结果中的第一张图片保存到本地磁盘。

    涉及 PIL 解码和磁盘写入（阻塞操作），请在工作线程中执行。

    Args:
        images: get_images 返回的 "节点 id -> 图片字节列表" 字典。

    Returns:
        保存到本地后的图片路径。
    """
    save_dir = r"D:/ComfyUI_AstrBot_Temp"
    os.makedirs(save_dir, exist_ok=True)

    for node_id in images:
        for i, image_data in enumerate(images[node_id]):
            image = Image.open(io.BytesIO(image_data))
            filename = f"node{node_id}_{i}.png"
            path = os.path.join(save_dir, filename)
            image.save(path)
            logger.info(f"saved: {path}")
            return f"D:/ComfyUI_AstrBot_Temp/{filename}"
@register("Ferrin's Toolkit", "Fylavvor", "神秘妙妙工具", "0.0.8a")
class MyPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""

    @filter.command_group("ff", alias={"f","鱼姐","肥鱼"})
    def ff():
        """基本功能模块"""
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

    @ff.command("picture", alias={"pic", "画图", "画画"})
    async def picture(self, event: AstrMessageEvent, user_prompt: str, aspect_ratio=3, mega_pixels=1.0):
        """调用本地ComfyUI生成图片"""
        global comfyui_queue

        if user_prompt == "help":
            yield event.plain_result("/ff picture (user_prompt: str) [aspect_ratio: int] [mega_pixels: float]\nuser_prompt：给模型的提示词（正面）。不要包含空格（因此建议用中文）。\naspect_ratio：图片宽高比，0-7分别对应：1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9。默认为3 (3:4)。\nmega_pixels：图片的总像素数（百万像素），最高2.0，最低0.5。默认为1.0。")
            return
        if function_picture==False:
            yield event.plain_result("该模块目前被禁用，请联系管理员。")
            return
        if comfyui_queue:
            yield event.plain_result("当前正在生成图片，请稍后再试。")
            return

        # 立即抢占队列标记（这中间没有 await，保证并发请求不会同时通过上面的检查）
        comfyui_queue = True
        umo = event.unified_msg_origin
        try:
            yield event.plain_result("尝试连接中...")

            prompt = json.loads(prompt_text, strict=False)
            # set the text prompt for our positive CLIPTextEncode
            prompt["2"]["inputs"]["text"] = user_prompt
            # set the seed for our KSampler node
            prompt["6"]["inputs"]["seed"] = random.randint(1, 2**32 - 1)

            # set the aspect ratio of the latent
            prompt["10"]["inputs"]["aspect_ratio"] = aspect_ratio_options[aspect_ratio]

            # set the pixels of the latent
            prompt["10"]["inputs"]["megapixels"] = max(min(2.0, mega_pixels), 0.5)

            ws = websocket.WebSocket()
            # 建连、等待生成、读取历史、保存图片都是阻塞操作，交给工作线程执行，
            # 否则会阻塞 AstrBot 的事件循环（生成期间其它消息将无法被处理）
            await asyncio.to_thread(
                ws.connect, "ws://{}/ws?clientId={}".format(server_address, client_id)
            )

            await self.context.send_message(umo, MessageChain().message(f"服务器连接成功，正在生成..."))
            try:
                image = await get_images(ws, prompt)
            finally:
                ws.close()

            path = await asyncio.to_thread(save, image)

            message_chain = MessageChain().file_image(f"{path}").message("图片已生成！")
            await self.context.send_message(umo, message_chain)
        finally:
            # 无论成功还是异常，都要释放队列标记，否则插件将再也无法生成图片
            comfyui_queue = False


    @filter.permission_type(filter.PermissionType.ADMIN)
    @ff.command("setip")
    async def setip(self,event:AstrMessageEvent, ip4: int):
        """修改ComfyUI所在ip"""
        global server_address
        server_address = f"192.168.1.{ip4}:8188"
        yield event.plain_result(f"ComfyUI服务器地址已修改：{server_address}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @ff.command("checkip", alias={"ip"})
    async def checkip(self,event:AstrMessageEvent):
        """查看当前存储的ComfyUI所在ip"""
        global server_address
        yield event.plain_result(f"当前ComfyUI服务器地址：{server_address}")

    @ff.command("dice", alias={"d"})
    async def dice(self, event: AstrMessageEvent, command: str, show_details=False):
        """扔骰子。可以指定数量和面数(显示简洁结果)。"""
        if command == "help":
            yield event.plain_result("/ff dice (command: str) 掷出骰子，只显示最终结果。\ncommand：指定投出骰子的数量与面数，格式：(dice_amount)d(dice_faces)。例如：3d6, 1d20。\nshow_details：是否显示详细结果。默认为False。")
            return
        dice_amount=None
        dice_face=None
        for index, char in enumerate(command):
            if char=='d':
                try:
                    dice_amount=int(command[:index])
                    dice_face=int(command[index+1:])
                except ValueError:
                    pass
                finally:
                    pass
                break
        die_result = []
        dice_result=0
        if dice_amount!=None and dice_face!=None:
            message=f"Throwing: {command}\nResults:"

            for i in range(dice_amount):
                die_result.append(random.randint(1,dice_face))
                dice_result+=die_result[i]
                message+=f" {die_result[i]}"

            message+=f"\nTotal: {dice_result}"

            yield event.plain_result(f"Result: {dice_result}")
        else:
            yield event.plain_result("输入格式错误！查阅/ff dice help以确定语法。")
                
    @ff.command("ddice", alias={"dd"})
    async def ddice(self, event: AstrMessageEvent, command: str, show_details=False):
        """扔骰子。可以指定数量和面数(显示详细结果)。"""
        if command == "help":
            yield event.plain_result("/ff ddice (command: str) 掷出骰子，并显示详细结果。\ncommand：指定投出骰子的数量与面数，格式：(dice_amount)d(dice_faces)。例如：3d6, 1d20。")
            return
        dice_amount=None
        dice_face=None
        for index, char in enumerate(command):
            if char=='d':
                try:
                    dice_amount=int(command[:index])
                    dice_face=int(command[index+1:])
                except ValueError:
                    pass
                finally:
                    pass
                break
            
        die_result = []
        dice_result=0

        if dice_amount!=None and dice_face!=None:
            message=f"Throwing: {command}\nResults:"

            for i in range(dice_amount):
                die_result.append(random.randint(1,dice_face))
                dice_result+=die_result[i]
                message+=f" {die_result[i]}"

            message+=f"\nTotal: {dice_result}"

            yield event.plain_result(message)

        else:
            yield event.plain_result("输入格式错误！查阅/ff ddice help以确定语法。")            

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("sw", alias={"switch"})
    async def sw(self, event: AstrMessageEvent, module: str, state="whatever"):
        if module == "help":
            yield event.plain_result("/sw (module: str) (state: str) 开启/关闭对应功能。")
            return        
        change_state: bool
        enable=["1", "true", "True", "TRUE", "Enable", "enable", "ENABLE", "on", "On", "ON"]
        disable=["0", "false", "False", "FALSE", "Disable", "disable", "DISABLE", "off", "Off", "OFF"]
        if state in enable:
            change_state=True
        elif state in disable:
            change_state=False
        else:
            yield event.plain_result("参数错误！请检查输入。")
            return
        if module=="pic" or module=="picture":
            global function_picture
            function_picture=change_state
            yield event.plain_result(f"ComfyUI Text-to-Image: {"enabled" if change_state else "disabled"}")
        else:
            yield event.plain_result("参数错误！请检查输入。")
            return
            

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
