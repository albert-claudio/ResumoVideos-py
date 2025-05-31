# 📼 Resumidor de Vídeos do YouTube com IA Generativa e PDF

Este script Python automatiza o processo de resumir vídeos do YouTube. Ele baixa o áudio do vídeo, transcreve-o para texto em português, utiliza uma IA generativa (API Gemini) para criar um resumo conciso e, finalmente, guarda esse resumo num ficheiro PDF.

---

## 🚀 Funcionalidades

- 🎵 **Download de Áudio**: Baixa automaticamente o áudio do vídeo do YouTube fornecido (utilizando `yt-dlp`).
- 🗣️ **Transcrição**: Converte o áudio para texto em português (utilizando `openai-whisper`).
- 🧠 **Sumarização com IA**: Envia a transcrição para a API Gemini para gerar um resumo inteligente e conciso.
- 📄 **Geração de PDF**: Cria um ficheiro PDF formatado com o resumo (utilizando `reportlab`).
- 🔐 **Configuração Segura da API Key**: Utiliza um ficheiro `.env` para guardar a chave da API Gemini de forma segura.

---

## ✅ Pré-requisitos

Certifique-se de que os seguintes itens estão instalados:

- **Python 3.7+** – [Download](https://www.python.org/downloads/)
- **pip** – Gestor de pacotes Python (geralmente vem com o Python)
- **FFmpeg** – Essencial para áudio/vídeo:

  - **Windows**: Baixe de [ffmpeg.org](https://ffmpeg.org/download.html) (versão "full build") e adicione o diretório `bin` ao PATH do sistema.
  - **macOS** (com Homebrew):  
    ```bash
    brew install ffmpeg
    ```
  - **Linux (Debian/Ubuntu)**:  
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

---

## ⚙️ Configuração

### 1. Clonar o Repositório (opcional)

```bash
git clone https://github.com/albert-claudio/ResumoVideos-py.git
cd seu-repositorio
````

### 2. Instalar Dependencias 
```bash
pip install yt-dlp openai-whisper torch torchaudio requests reportlab python-dotenv
````

### 3. Configurar a Chave da API Gemini
```bash
GEMINI_API_KEY=SUA_CHAVE_API_AQUI
```
- **Importante**: Adicione `.env` ao `.gitignore` para proteger sua chave da API.

### 4. Fonte para PDF(opcional)
Para melhor renderização de caracteres acentuados, baixe a fonte DejaVuSans.ttf e coloque-a no mesmo diretório do script.
[Download da Fonte](https://dejavu-fonts.github.io/)
