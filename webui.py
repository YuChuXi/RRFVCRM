import gradio as gr



title_0 = "RRFVCRM"
title_1 = "Thanks for doctor Bo Peng created the RWKV model!"
title_2 = "Powered By AMD Radeon Pro W7900!"



with gr.Blocks(title=title_0) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title_1}</h1>\n</div>")
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title_2}</h1>\n</div>")
    with gr.Tab("Chat and generate speech, facial movements and actions"):
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
    with gr.Tab("Generate novel"):           ##text model tab
        gr.Markdown(f"Thanks for doctor Bo peng created the RWKV model!")
        gr.Markdown(f"######玉子姐姐最可爱了～～～######")
        gr.Markdown(f"######模型被调教坏了从我显卡上滚出去！要被玩坏的～～～######")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Prompt", value="")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=5)
        #submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
    with gr.Tab("RWKV Music"):
        gr.Markdown(f"Thanks for doctor Bo Peng created the RWKV model!")
    with gr.Tab("Visual RWKV"):
        gr.Markdown(f"Thanks for doctor Bo Peng created the RWKV model!")
        with gr.Row():
            with gr.Column():
                input_picture = gr.Image(label="input picture")
                input = gr.Textbox(lines=2, label="input", value="")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=5)
    with gr.Tab("Pure chat"):           ##text model tab
        gr.Markdown(f"######玉子姐姐最可爱了～～～######")
        gr.Markdown(f"######模型被调教坏了从我显卡上滚出去！要被玩坏的～～～######")
        with gr.Row():
            with gr.Column():
                input = gr.Textbox(lines=2, label="input", value="")
                user_name = gr.Textbox(lines=1,label="Pleas press you user name~", value="")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 10.0, label="Presence Penalty", step=0.1, value=1)
                count_penalty = gr.Slider(0.0, 10.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")                    
                output = gr.Textbox(label="Output", lines=5)               
        #submit.click(chat, [user_name,input, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
    with gr.Tab("Announce Tab"):
        gr.Markdown(f"Thanks for doctor Bo Peng created the RWKV model!")
     
demo.queue(default_concurrency_limit=6)   #多线程设置
demo.launch(server_name="192.168.0.105", server_port=11451, show_error=True, share=False)