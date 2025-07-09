import gradio as gr
import os
from groq import Groq

# Função principal que interage com a API da GROQ
def get_groq_response(message, history, api_key, system_prompt, temperature):
    """
    Obtém uma resposta da API da GROQ.
    A função é um gerador para suportar streaming de texto.
    """
    final_api_key = api_key or os.environ.get("GROQ_API_KEY")

    if not final_api_key:
        yield "Erro: A chave da API da GROQ não foi informada. Por favor, insira a chave na interface ou defina a variável de ambiente GROQ_API_KEY."
        return

    client = Groq(api_key=final_api_key)

    # Formata o histórico para o formato da API, incluindo as instruções do sistema
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        # Cria a chamada de chat com streaming
        stream = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=temperature,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )

        # Yield cada parte da resposta para o Gradio
        response_text = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            response_text += content
            yield response_text

    except Exception as e:
        yield f"Ocorreu um erro: {e}"

# Construção da interface com Gradio Blocks
with gr.Blocks(theme=gr.themes.Soft(), title="Chatbot Jurídico com GROQ") as demo:
    gr.Markdown(
        """
        # Chatbot Jurídico com GROQ
        Faça sua pergunta e o chatbot irá te auxiliar.
        Ajuste as configurações e informe sua chave da API da GROQ.
        """
    )

    with gr.Accordion("Configurações Avançadas", open=False):
        system_prompt = gr.Textbox(
            label="Instruções do Sistema",
            value="Você é um assistente jurídico prestativo e preciso, que sempre responde em português do Brasil.",
            info="Defina o comportamento do chatbot.",
            lines=3,
        )
        temperature = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            step=0.1,
            value=0.7,
            label="Temperatura",
            info="Controla a criatividade da resposta. Valores mais altos geram respostas mais criativas.",
        )
        groq_api_key = gr.Textbox(
            label="Chave da API da GROQ",
            type="password",
            placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            info="Insira sua chave da API da GROQ aqui. Se deixado em branco, usará a variável de ambiente GROQ_API_KEY.",
        )

    # Interface de Chat
    chat_interface = gr.ChatInterface(
        fn=get_groq_response,
        additional_inputs=[groq_api_key, system_prompt, temperature],
        title="Assistente Jurídico",
        description="Este chatbot usa o modelo Llama3 via GROQ para responder suas perguntas.",
        examples=[
            ["O que é um habeas corpus?"],
            ["Quais são os tipos de divórcio existentes no Brasil?"],
            ["Explique o que é usucapião de forma simples."]
        ]
    )

if __name__ == "__main__":
    # Inicia a aplicação Gradio
    demo.launch()
