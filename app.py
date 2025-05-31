#Resumir Vídeos do YouTube
#
# Funcionalidades:
# 1. Baixa o áudio de um vídeo do YouTube usando yt-dlp.
# 2. Transcreve o áudio para texto usando o Whisper.
# 3. Resume o texto usando um modelo da Hugging Face Transformers.
#
# Requisitos (instale via pip):
# pip install yt-dlp openai-whisper transformers torch torchaudio
#
# Requisito Adicional (Software Externo):
# - FFmpeg: Necessário para o processamento de áudio pelo yt-dlp e Whisper.
#   (Instruções de instalação no script original ou na internet)
#
# Instruções de Uso:
# 1. Certifique-se de que todas as bibliotecas, yt-dlp e FFmpeg estão instalados.
# 2. Execute o script.
# 3. Quando solicitado, insira a URL completa do vídeo do YouTube.
# 4. Aguarde o processamento.
import os
import tempfile
import subprocess
import json # Para trabalhar com o payload e a resposta da API
import requests # Para fazer chamadas HTTP à API Gemini
import re # Para limpar o nome do ficheiro
import whisper
import torch # Whisper pode depender de torch

from dotenv import load_dotenv

# Importações para ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

def limpar_nome_ficheiro(url):
    """
    Cria um nome de ficheiro seguro a partir de uma URL ou título.
    """
    try:
        # Tenta extrair o ID do vídeo para um nome mais curto
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if match:
            nome_base = match.group(1)
        else:
            # Se não for uma URL do YouTube reconhecível, usa uma parte da URL
            nome_base = url.split('/')[-1] if '/' in url else url
            nome_base = nome_base.split('?')[0] # Remove parâmetros query
        
        # Remove caracteres inválidos para nomes de ficheiro
        nome_seguro = re.sub(r'[\\/*?:"<>|]', "", nome_base)
        nome_seguro = nome_seguro[:50] # Limita o comprimento
        return f"resumo_video_{nome_seguro}"
    except Exception:
        return "resumo_video_desconhecido"


def baixar_audio_youtube_yt_dlp(url, output_dir):
    """
    Baixa a melhor stream de áudio de um vídeo do YouTube usando yt-dlp.
    Retorna o caminho do ficheiro de áudio baixado ou None em caso de erro.
    """
    try:
        print(f"A tentar baixar áudio com yt-dlp de: {url}...")
        try:
            subprocess.run(['yt-dlp', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\nERRO: yt-dlp não foi encontrado ou não está a funcionar corretamente.")
            print("Por favor, instale ou verifique a sua instalação de yt-dlp: 'pip install yt-dlp'")
            print("Certifique-se também de que está no PATH do seu sistema.\n")
            return None

        audio_format = 'm4a'
        audio_filename_base = "downloaded_audio_for_summary"
        output_filepath = os.path.join(output_dir, f"{audio_filename_base}.{audio_format}")

        command = [
            'yt-dlp', '--extract-audio', '-x', '--audio-format', audio_format,
            '--output', output_filepath, '--no-playlist', '--quiet', '--no-warnings',
            url
        ]
        
        print(f"A executar comando: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            if os.path.exists(output_filepath):
                print(f"Áudio baixado com yt-dlp para: {output_filepath}")
                return output_filepath
            else:
                print(f"yt-dlp executado com sucesso, mas o ficheiro de áudio esperado ({output_filepath}) não foi encontrado.")
                print(f"Verifique o conteúdo do diretório temporário: {output_dir}")
                print(f"Stdout: {stdout.decode('utf-8', errors='ignore')}")
                print(f"Stderr: {stderr.decode('utf-8', errors='ignore')}")
                possible_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(f'.{audio_format}')]
                if possible_files:
                    found_file = possible_files[0]
                    print(f"Ficheiro de áudio alternativo encontrado: {found_file}")
                    return found_file
                return None
        else:
            print(f"Erro ao baixar o áudio com yt-dlp (código de saída: {process.returncode}):")
            error_message = stderr.decode('utf-8', errors='ignore').strip() or stdout.decode('utf-8', errors='ignore').strip()
            print(f"Mensagem de erro do yt-dlp: {error_message if error_message else 'Nenhuma mensagem de erro específica.'}")
            return None
    except FileNotFoundError:
        print("ERRO CRÍTICO: O executável 'yt-dlp' não foi encontrado.")
        print("Certifique-se de que o yt-dlp está instalado ('pip install yt-dlp') e acessível no PATH do seu sistema.")
        return None
    except Exception as e:
        print(f"Exceção inesperada ao tentar usar yt-dlp: {e}")
        return None

def transcrever_audio(caminho_audio, modelo_whisper="base"):
    """
    Transcreve o áudio para texto em Português usando o modelo Whisper.
    Retorna o texto transcrito ou None em caso de erro.
    """
    if not caminho_audio or not os.path.exists(caminho_audio):
        print(f"Erro na transcrição: Caminho do áudio não encontrado ou inválido: {caminho_audio}")
        return None
    try:
        print(f"A carregar modelo Whisper ({modelo_whisper})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"A usar dispositivo para Whisper: {device}")
        
        model = whisper.load_model(modelo_whisper, device=device)
        print("Modelo Whisper carregado. A transcrever áudio em Português...")
        
        resultado = model.transcribe(caminho_audio, fp16=torch.cuda.is_available(), language="pt")
        texto_transcrito = resultado["text"]
        
        print("Transcrição concluída.")
        if not texto_transcrito.strip():
            print("A transcrição resultou em texto vazio.")
            return None
        return texto_transcrito
    except Exception as e:
        print(f"Erro durante a transcrição do áudio: {e}")
        return None

def resumir_texto_com_gemini(texto_transcrito, max_tokens_resumo=250):
    """
    Resume o texto transcrito usando a API Gemini.
    Retorna o texto resumido ou None em caso de erro.
    """
    if not texto_transcrito or not texto_transcrito.strip():
        print("Nenhum texto para resumir.")
        return None

    print("A iniciar sumarização com a API Gemini...")
    
    api_key = os.getenv("GEMINI_API_KEY") # Certifique-se de que a chave da API está definida no .env
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}" 

    prompt_instrucao = (
        f"Você é um assistente especializado em resumir textos de forma concisa e precisa em português brasileiro.\n"
        f"Resuma o seguinte texto, extraindo as informações mais importantes. "
        f"Tente manter o resumo com aproximadamente {int(max_tokens_resumo * 0.6)} a {max_tokens_resumo} palavras, " # Ajustado para um intervalo mais amplo
        f"mas priorize a qualidade e a cobertura dos pontos chave sobre a contagem exata de palavras.\n\n"
        f"Texto a ser resumido:\n\"\"\"\n{texto_transcrito}\n\"\"\"" 
    ) 

    chat_history = [{"role": "user", "parts": [{"text": prompt_instrucao}]}]
    payload = {"contents": chat_history}
    
    headers = {'Content-Type': 'application/json'}

    try: # Aumentar o timeout para 180 segundos para evitar problemas com vídeos longos
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=180) # Timeout aumentado para 180s
        response.raise_for_status() 
        
        result = response.json()

        if (result.get("candidates") and 
            result["candidates"][0].get("content") and
            result["candidates"][0]["content"].get("parts") and
            result["candidates"][0]["content"]["parts"][0].get("text")):
            
            texto_resumido = result["candidates"][0]["content"]["parts"][0]["text"]
            print("Resumo gerado pela API Gemini.")
            return texto_resumido.strip()
        else:
            print("Erro: A resposta da API Gemini não continha o texto do resumo esperado.")
            print(f"Resposta completa da API: {result}")
            if result.get("promptFeedback"):
                print(f"Feedback do prompt: {result.get('promptFeedback')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição à API Gemini: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Detalhes do erro da API: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Conteúdo da resposta (não JSON): {e.response.text}")
        return None
    except Exception as e:
        print(f"Erro inesperado ao processar resposta da API Gemini: {e}")
        return None

def gerar_pdf_resumo(texto_resumo, nome_ficheiro_pdf):
    """
    Gera um ficheiro PDF com o texto do resumo.
    """
    if not texto_resumo or not texto_resumo.strip():
        print("Nenhum texto de resumo para gerar PDF.")
        return False

    try:
        print(f"A gerar PDF: {nome_ficheiro_pdf}...")
        
        # Tenta registar uma fonte que suporte caracteres portugueses comuns
        # Se 'DejaVuSans' não estiver disponível, o reportlab usará uma fonte padrão.
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            font_name = 'DejaVuSans'
            print("Fonte 'DejaVuSans' registada para o PDF.")
        except Exception: # Erro comum se a fonte não estiver instalada
            font_name = 'Helvetica' # Fonte padrão do ReportLab
            print("A usar fonte padrão 'Helvetica' para o PDF (DejaVuSans não encontrada).")


        doc = SimpleDocTemplate(nome_ficheiro_pdf, pagesize=A4,
                                rightMargin=inch, leftMargin=inch,
                                topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        
        # Estilo para o título
        style_titulo = styles['h1']
        style_titulo.fontName = font_name
        style_titulo.alignment = 1 # Centralizado
        
        # Estilo para o corpo do texto
        style_corpo = styles['Normal']
        style_corpo.fontName = font_name
        style_corpo.fontSize = 11
        style_corpo.leading = 14 # Espaçamento entre linhas
        style_corpo.alignment = 4 # Justificado

        story = []
        
        titulo_texto = "Resumo do Vídeo do YouTube"
        paragrafo_titulo = Paragraph(titulo_texto, style_titulo)
        story.append(paragrafo_titulo)
        story.append(Spacer(1, 0.3*inch))

        # Substituir quebras de linha por tags <br/> para o Paragraph do ReportLab
        texto_resumo_formatado = texto_resumo.replace('\n', '<br/>\n')
        paragrafo_resumo = Paragraph(texto_resumo_formatado, style_corpo)
        story.append(paragrafo_resumo)
        
        doc.build(story)
        print(f"PDF gerado com sucesso: {nome_ficheiro_pdf}")
        return True
    except FileNotFoundError as e:
        print(f"Erro ao gerar PDF: Ficheiro de fonte não encontrado. {e}")
        print("Certifique-se de que a fonte (ex: DejaVuSans.ttf) está no mesmo diretório do script ou instale-a no sistema.")
        print("O script tentou usar 'Helvetica' como fallback, mas pode haver outros problemas.")
        return False
    except Exception as e:
        print(f"Erro ao gerar PDF: {e}")
        return False

def main():
    print("--- Resumidor de Vídeos do YouTube (com IA Generativa Gemini e PDF) ---")
    
    if os.system("ffmpeg -version > nul 2>&1" if os.name == 'nt' else "ffmpeg -version > /dev/null 2>&1") != 0:
        print("\nAVISO: FFmpeg não encontrado. Essencial para yt-dlp e Whisper.")
        print("Instale o FFmpeg e adicione-o ao PATH.\n")

    url_video = input("Insira a URL do vídeo do YouTube: ")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"A usar diretório temporário: {tmpdir}")
        caminho_audio = baixar_audio_youtube_yt_dlp(url_video, tmpdir)

        if caminho_audio:
            texto_transcrito = transcrever_audio(caminho_audio, modelo_whisper="base")
            
            if texto_transcrito:
                print("\n--- Texto Transcrito Completo ---") 
                print(texto_transcrito) 
                
                print("\n--- A iniciar Sumarização com IA Generativa ---") 
                texto_resumido = resumir_texto_com_gemini(texto_transcrito, max_tokens_resumo=200) 
                
                if texto_resumido:
                    print("\n--- Resumo do Vídeo (Gerado por IA) ---")
                    print(texto_resumido)

                    # Gerar PDF com o resumo
                    nome_base_ficheiro_pdf = limpar_nome_ficheiro(url_video)
                    nome_ficheiro_pdf_final = f"{nome_base_ficheiro_pdf}.pdf"
                    
                    if gerar_pdf_resumo(texto_resumido, nome_ficheiro_pdf_final):
                        print(f"O resumo também foi guardado em: {os.path.abspath(nome_ficheiro_pdf_final)}")
                    else:
                        print("Falha ao gerar o ficheiro PDF com o resumo.")
                else:
                    print("Não foi possível gerar o resumo utilizando a IA Generativa.")
            else:
                print("Não foi possível transcrever o áudio.")
        else:
            print("Não foi possível baixar o áudio do vídeo usando yt-dlp.")

    print("\n--- Processo Concluído ---")

if __name__ == "__main__":
    main()