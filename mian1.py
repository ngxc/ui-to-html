from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import requests
import base64

app = Flask(__name__)
CORS(app)

# ========== 工具：图片转 Base64 ==========
def img_to_base64(file):
    return base64.b64encode(file.read()).decode()

# ========== 调用 SiliconFlow 图像模型 ==========
def call_siliconflow_with_image(api_key, prompt, user_text, image_base64,
                                model="Qwen/Qwen3-VL-32B-Thinking",
                                temperature=0.15, top_p=0.3, top_k=50):
    API_URL = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user",
             "content": [
                 {"type": "text", "text": user_text},
                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
             ]}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ========== 调用 SiliconFlow 文本模型 ==========
def call_siliconflow_text(api_key, text,
                          model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
                          temperature=0.7, max_tokens=8048):
    API_URL = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ==================== 内嵌 HTML ====================
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>UI → HTML 生成工具</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
body {margin:0; padding:0; font-family:'Poppins',sans-serif; background:linear-gradient(135deg,#1f1c2c,#928dab); height:100vh; display:flex; justify-content:center; align-items:flex-start; overflow-y:auto; color:#fff;}
.container {width:90%; max-width:900px; margin-top:40px; padding:30px 40px; background:rgba(255,255,255,0.08); border-radius:18px; backdrop-filter:blur(18px); box-shadow:0 8px 28px rgba(0,0,0,0.35); border:1px solid rgba(255,255,255,0.2);}
h2,h3{font-weight:600;margin-bottom:15px;}
label{display:block;margin-top:20px;font-size:15px;opacity:0.9;}
input[type="password"], input[type="file"], input[type="text"], select{margin-top:8px; width:100%; padding:12px 16px; border-radius:12px; border:1px solid rgba(255,255,255,0.3); background:rgba(255,255,255,0.15); color:#fff; font-size:15px; outline:none;}
button{margin-top:25px; width:100%; padding:14px; font-size:16px; border:none; border-radius:12px; background:linear-gradient(135deg,#00c6ff,#0072ff); color:#fff; cursor:pointer; font-weight:600; transition:0.25s;}
button:hover{transform:translateY(-2px); box-shadow:0 8px 20px rgba(0,122,255,0.5);}
#status{margin-top:15px;font-size:15px;font-weight:500;}
pre{padding:18px; background:rgba(0,0,0,0.4); border-radius:12px; white-space:pre-wrap; word-wrap:break-word; border:1px solid rgba(255,255,255,0.1);}
.param-row{display:flex; gap:10px; margin-top:10px;}
.param-row input{flex:1;}
</style>
</head>
<body>
<div class="container">
<h2>UI → HTML 自动生成工具</h2>
<label>SiliconFlow API Key：</label>
<input id="api_key" type="password" placeholder="sk-xxxx">
<label>上传 UI 图片：</label>
<input id="image" type="file" accept="image/*">
<label>选择 VLM 模型：</label>
<select id="vlm_model">
<option value="Qwen/Qwen3-VL-32B-Thinking">Qwen/Qwen3-VL-32B-Thinking</option>
<option value="Qwen/Qwen3-VL-30B-A3B-Instruct">Qwen/Qwen3-VL-30B-A3B-Instruct</option>
<option value="Qwen/Qwen3-VL-235B-A22B-Instruct">Qwen/Qwen3-VL-235B-A22B-Instruct</option>
<option value="Qwen/Qwen3-VL-235B-A22B-Thinking">Qwen/Qwen3-VL-235B-A22B-Thinking</option>
<option value="Qwen/Qwen3-Omni-30B-A3B-Captioner">Qwen/Qwen3-Omni-30B-A3B-Captioner</option>
</select>

<div class="param-row">
<input id="vlm_temperature" type="number" step="0.05" min="0" max="2" placeholder="VLM temperature (默认0.15)">
<input id="vlm_top_p" type="number" step="0.05" min="0" max="1" placeholder="VLM top_p (默认0.3)">
<input id="vlm_top_k" type="number" step="1" min="1" placeholder="VLM top_k (默认50)">
</div>

<label>选择代码模型：</label>
<select id="code_model">
<option value="Qwen/Qwen3-Coder-480B-A35B-Instruct">Qwen/Qwen3-Coder-480B-A35B-Instruct</option>
<option value="Qwen/Qwen3-Coder-30B-A3B-Instruct">Qwen/Qwen3-Coder-30B-A3B-Instruct</option>
<option value="moonshotai/Kimi-K2-Instruct-0905">moonshotai/Kimi-K2-Instruct-0905</option>
</select>
<div class="param-row">
<input id="code_temperature" type="number" step="0.05" min="0" max="2" placeholder="Code temperature (默认0.7)">
</div>

<button onclick="submitForm()">开始生成</button>
<p id="status"></p>
<h3>UI 结构化描述</h3>
<pre id="ui_desc"></pre>
<h3>生成的 HTML 代码</h3>
<pre id="html_code"></pre>

<h3>代码模型聊天 / 修改 HTML</h3>
<label>输入指令：</label>
<input id="code_instruction" type="text" placeholder="例如：把按钮颜色改成红色">
<button onclick="sendCodeInstruction()">提交指令</button>
<pre id="html_chat_response"></pre>

<script>
async function submitForm() {
    const api_key = document.getElementById("api_key").value;
    const image = document.getElementById("image").files[0];
    const vlm_model = document.getElementById("vlm_model").value;
    const code_model = document.getElementById("code_model").value;
    const vlm_temperature = parseFloat(document.getElementById("vlm_temperature").value)||0.15;
    const vlm_top_p = parseFloat(document.getElementById("vlm_top_p").value)||0.3;
    const vlm_top_k = parseInt(document.getElementById("vlm_top_k").value)||50;
    const code_temperature = parseFloat(document.getElementById("code_temperature").value)||0.7;

    const status = document.getElementById("status");
    if (!api_key || !image) { alert("请填写 API Key 并上传图片"); return; }
    status.innerText = "⏳ 正在生成 UI 描述...";
    const formData1 = new FormData();
    formData1.append("api_key", api_key);
    formData1.append("image", image);
    formData1.append("vlm_model", vlm_model);
    formData1.append("vlm_temperature", vlm_temperature);
    formData1.append("vlm_top_p", vlm_top_p);
    formData1.append("vlm_top_k", vlm_top_k);

    const descResp = await fetch("/generate_description", { method:"POST", body:formData1 });
    const descData = await descResp.json();
    if(descData.error){ status.innerText="❌ 错误："+descData.error; return; }
    document.getElementById("ui_desc").innerText = descData.ui_description;

    status.innerText = "⏳ UI 描述完成，正在生成 HTML 代码...";
    const formData2 = new FormData();
    formData2.append("api_key", api_key);
    formData2.append("ui_description", descData.ui_description);
    formData2.append("code_model", code_model);
    formData2.append("code_temperature", code_temperature);

    const htmlResp = await fetch("/generate_html", { method:"POST", body:formData2 });
    const htmlData = await htmlResp.json();
    if(htmlData.error){ status.innerText="❌ 错误："+htmlData.error; return; }
    document.getElementById("html_code").innerText = htmlData.html_code;
    status.innerText = "✔ 全部完成！";
}

async function sendCodeInstruction() {
    const api_key = document.getElementById("api_key").value;
    const code_model = document.getElementById("code_model").value;
    const ui_html = document.getElementById("html_code").innerText;
    const instruction = document.getElementById("code_instruction").value;
    const code_temperature = parseFloat(document.getElementById("code_temperature").value)||0.7;
    const status = document.getElementById("status");

    if(!api_key || !ui_html || !instruction){ alert("请确保 API Key、HTML 和指令均已填写"); return; }

    status.innerText = "⏳ 代码模型处理中...";
    const formData = new FormData();
    formData.append("api_key", api_key);
    formData.append("code_model", code_model);
    formData.append("html_code", ui_html);
    formData.append("instruction", instruction);
    formData.append("code_temperature", code_temperature);

    const resp = await fetch("/code_model_chat", { method:"POST", body:formData });
    const data = await resp.json();
    if(data.error){ status.innerText="❌ 错误："+data.error; return; }

    document.getElementById("html_chat_response").innerText = data.updated_html;
    status.innerText = "✔ 指令执行完成！";
}
</script>
</div>
</body>
</html>"""

# ==================== Flask 路由 ====================
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/generate_description", methods=["POST"])
def generate_description():
    try:
        api_key = request.form.get("api_key")
        vlm_model = request.form.get("vlm_model")
        file = request.files.get("image")
        temperature = float(request.form.get("vlm_temperature") or 0.15)
        top_p = float(request.form.get("vlm_top_p") or 0.3)
        top_k = int(request.form.get("vlm_top_k") or 50)

        if not api_key: return jsonify({"error": "缺少 API Key"}), 400
        if not file: return jsonify({"error": "请上传图片"}), 400
        if not vlm_model: vlm_model = "Qwen/Qwen3-VL-32B-Thinking"

        img_b64 = img_to_base64(file)
        ui_prompt = (
            "请以专业详细、严谨且准确的方式，对该界面的 UI 设计进行结构化描述。"
            "要求详细明确说明界面中各主要功能区域的布局具体位置、层级关系、空间划分、交互元素分布等关键细节，"
            "如果有图片的话就说有图片预留位置，"
            "不允许有额外主观推测，"
            "并使用标准化的界面设计术语进行描述。"
        )
        ui_desc = call_siliconflow_with_image(api_key, ui_prompt, "", img_b64,
                                              model=vlm_model,
                                              temperature=temperature,
                                              top_p=top_p,
                                              top_k=top_k)
        return jsonify({"ui_description": ui_desc, "image_base64": img_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_html", methods=["POST"])
def generate_html():
    try:
        api_key = request.form.get("api_key")
        ui_description = request.form.get("ui_description")
        code_model = request.form.get("code_model")
        temperature = float(request.form.get("code_temperature") or 0.7)

        if not api_key or not ui_description: return jsonify({"error": "缺少参数"}), 400
        if not code_model: code_model = "Qwen/Qwen3-Coder-480B-A35B-Instruct"

        html_prompt = f"请根据以下 UI 描述生成对应 HTML 代码（只输出代码，不要解释）：\n{ui_description}"
        html_code = call_siliconflow_text(api_key, html_prompt, model=code_model, temperature=temperature)
        return jsonify({"html_code": html_code})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/code_model_chat", methods=["POST"])
def code_model_chat():
    try:
        api_key = request.form.get("api_key")
        code_model = request.form.get("code_model")
        html_code = request.form.get("html_code")
        instruction = request.form.get("instruction")
        temperature = float(request.form.get("code_temperature") or 0.7)

        if not api_key or not html_code or not instruction:
            return jsonify({"error": "缺少参数"}), 400
        if not code_model: code_model = "Qwen/Qwen3-Coder-480B-A35B-Instruct"

        prompt = f"根据下面 HTML 代码和用户指令生成修改后的 HTML 代码（只输出代码，不要解释）：\n\nHTML:\n{html_code}\n\n指令:\n{instruction}"
        updated_html = call_siliconflow_text(api_key, prompt, model=code_model, temperature=temperature)
        return jsonify({"updated_html": updated_html})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✔ Flask 已启动：http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
