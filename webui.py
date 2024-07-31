import gradio as gr
import os
from models.rwkv6.dialogue import chat, evaluate
from models.visualRWKV.app.app_gpu import chatbot_v

title_0 = "RRFVCRM"
title_1 = "Thanks for doctor Bo Peng created the RWKV model!"
title_2 = "Powered By AMD Radeon Pro W7900!"
title_3 = "It is not Available now"

cur_dir = os.path.dirname(os.path.abspath(__file__))
examples = [
    [
        f"{cur_dir}/models/visualRWKV/app/examples_pizza.jpg",
        "What are steps to cook it?"
    ],
    [
        f"{cur_dir}/models/visualRWKV/app/examples_bluejay.jpg",
        "what is the name of this bird?",
    ],
    [
        f"{cur_dir}/models/visualRWKV/app/examples_extreme_ironing.jpg",
        "What is unusual about this image?",
    ],
    [
        f"{cur_dir}/models/visualRWKV/app/examples_waterview.jpg",
        "What are the things I should be cautious about when I visit here?",
    ],
]


with gr.Blocks(title=title_0) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title_0}</h1>\n</div>")
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title_2}</h1>\n</div>")
    
    
    with gr.Tab("Chat and generate speech"):        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=12, label="Input", value="")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0.3)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.3)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")            
                output = gr.Textbox(label="Text Output", lines=12)
                voice_output = gr.Audio(value = None , format = "wave" , label = "Voice Output" )
        submit.click(chat, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
    
    
    with gr.Tab("Generate novel"):           ##text model tab        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=10, label="Prompt", value="")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=25)
        submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
    
    
    with gr.Tab("Chat and generate speech, facial movements and actions"):
        gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title_3}</h1>\n</div>")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Input", value="")
                voice_imput = gr.Audio(sources= ["microphone"] , format = "wave" , label = "Voice Input")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                face_motion = gr.Textbox(label="Face Motion Command Output", lines=1)
                action = gr.Textbox(label="Action Command Output", lines=1)
                output = gr.Textbox(label="Text Output", lines=5)
                voice_output = gr.Audio(value = None , format = "wave" , label = "Voice Output" )
        #submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
    
    
    with gr.Tab("RWKV Music"):        
        gr.Markdown(f"                     You can use the demo in ./model/music/")
        gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title_3}</h1>\n</div>")
    
    
    with gr.Tab("Visual RWKV"):            
        with gr.Row():
            with gr.Column():
                image = gr.Image(type='pil', label="Image")
            with gr.Column():
                prompt = gr.Textbox(lines=8, label="Prompt", 
                    value="Render a clear and concise summary of the photo.")
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary") 
            with gr.Column():
                output = gr.Textbox(label="Output", lines=10)
        data = gr.Dataset(components=[image, prompt], samples=examples, label="Examples", headers=["Image", "Prompt"])
        submit.click(chatbot_v, [image, prompt], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [image, prompt])

    
    
    with gr.Tab("chat bot"): ##chat bot model tab
        gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title_3}</h1>\n</div>")
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="Chatbot", height=500)
                prompt = gr.Textbox(lines=2, label="", value="")
                submit = gr.Button("Submit", variant="primary")
                clear = gr.ClearButton([prompt, chatbot])
            with gr.Column():
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
        #submit.click(chat, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
    
    with gr.Tab("Announce Tab"):
        gr.Markdown(f"Thanks for doctor Bo Peng created the RWKV model!")
        gr.Markdown(f"[RWKV-LM](https://github.com/BlinkDL/RWKV-LM)")
        gr.Markdown(f"[SoftVC VITS Singing Conversion](https://github.com/justinjohn0306/so-vits-svc-4.0/tree/4.0-v2)")
        gr.Markdown(f"[GPT-SoVITS-WebUI](https://github.com/RVC-Boss/GPT-SoVITS)")
        gr.Markdown(f"[RMVPE](https://github.com/Dream-High/RMVPE)")
        gr.Markdown(f"[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)")
        gr.Markdown(f"[OpenSeeFace](https://github.com/emilianavt/OpenSeeFace)")
        gr.Markdown(f"[DeepSpeech](https://github.com/mozilla/DeepSpeech)")
        gr.Markdown(f"[Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN)")
        gr.Markdown(f"[CelebV-Text](https://github.com/celebv-text/CelebV-Text)")
        gr.Markdown(f"Radeon Pro w7900 provided by [AMD](https://amd.com) ")


demo.queue()       
demo.launch(server_name="127.0.0.1", server_port=8080, show_error=True, share=False)
