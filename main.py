import torch
import PIL.Image as Image
import os
import cv2
import re
 
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers import LogitsProcessorList, NoBadWordsLogitsProcessor, AutoTokenizer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime


def calculate_statistics(results):
    """
    Calcula estadísticas de los resultados
    
    Args:
        results (list): Lista de diccionarios con los resultados
        
    Returns:
        dict: Diccionario con las estadísticas
    """
    total = len(results)
    correct = sum(1 for r in results if r['matches'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    errors = []
    for r in results:
        if not r['matches']:
            errors.append({
                'file': r['file'],
                'expected': r['expected'],
                'predicted': r['predicted']
            })
    
    return {
        'total_images': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'errors': errors
    }

def add_statistics_to_pdf(story, stats, styles):
    """
    Añade la sección de estadísticas al PDF
    """
    story.append(Spacer(1, 20))
    story.append(Paragraph("Estadísticas de Inferencia", styles['Heading2']))
    
    stats_text = [
        f"Total de imágenes procesadas: {stats['total_images']}",
        f"Predicciones correctas: {stats['correct_predictions']}",
        f"Precisión: {stats['accuracy']:.2f}%"
    ]
    
    for text in stats_text:
        story.append(Paragraph(text, styles['Normal']))
    
    if stats['errors']:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Errores encontrados:", styles['Heading3']))
        for error in stats['errors']:
            error_text = f"Archivo: {error['file']} - Esperado: {error['expected']}, Predicho: {error['predicted']}"
            story.append(Paragraph(error_text, styles['RedText']))

def process_image(image_path, target_size=(448, 448)):
    """
    Procesa una imagen para ser usada con el modelo PaLiGeMMA.
    
    Args:
        image_path (str): Ruta al archivo de imagen
        target_size (tuple): Tamaño objetivo de la imagen (ancho, alto)
    
    Returns:
        PIL.Image: Imagen procesada
    """
    try:
      
        image = cv2.imread(image_path)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_resized = pil_image.resize(target_size)
        return image_resized
      
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {str(e)}")
        raise

def generate_pdf():
    # configuracion  del PDF en modo paisaje para aprovechar mejor el espacio
    pdf_name = f"identificadores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(pdf_name, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    story = []

    styles.add(ParagraphStyle(
        name='GreenText',
        parent=styles['Normal'],
        textColor=colors.green
    ))
    styles.add(ParagraphStyle(
        name='RedText',
        parent=styles['Normal'],
        textColor=colors.red
    ))

    # configuración para inferencias.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "../models/paligemma2-3b-pt-448"
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    ).eval()

    processor = PaliGemmaProcessor.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    images_folder = "images"
    results = []
    
    forbidden_numbers = ['797', '930'] #en ocasiones el resultado es el modelo del camion minero de la fotografia: ejemplo: cat 797 o komatsu 930
    bad_words_ids = [tokenizer.encode(num, add_special_tokens=False) for num in forbidden_numbers]

    # procesador de logits
    logits_processor = LogitsProcessorList([
        NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids, eos_token_id=tokenizer.eos_token_id)
    ])

    for image_file in os.listdir(images_folder)[:18]:
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, image_file)
            
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_folder, image_file)
            
            image = process_image(image_path, target_size=(448, 448))
            if image is None:
                print(image_path, " sin inferencia")
                continue
            
            prompt = """User: Look at this mining truck and tell me only the single largest number painted on its side. 
            If you see multiple numbers, only return the most prominent large ID number.
            Respond ONLY with the format: Number: <single_number>
            Do not include any letters, prefixes, or additional numbers.
            <image>
            Assistant: """
            model_inputs = processor(
                text=prompt, 
                images=image, 
                padding="longest", 
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = model.generate(
                    **model_inputs,
                    max_new_tokens=64,
                    num_beams=7,
                    logits_processor=logits_processor,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0,
                    early_stopping=True
                )
            
            prediction = process_model_output(output, len(prompt), processor)
            expected_number = re.search(r'\d+', image_file).group()
          
            matches = expected_number == prediction
            print("expected_number", expected_number, "prediction", prediction)
            results.append({
                'file': image_file,
                'path': image_path,
                'expected': expected_number,
                'predicted': prediction,
                'matches': matches
            })
            
            del model_inputs
            del output
            torch.cuda.empty_cache()

    stats = calculate_statistics(results)
    
    # Crear tabla de imágenes (6 columnas)
    table_data = []
    current_row = []
    
    for result in results:
        cell_content = []
        cell_content.append(RLImage(result['path'], width=75, height=55))
        cell_content.append(Paragraph(
            f"Esperado: {result['expected']}<br/>Resultado: {result['predicted']}",
            styles["GreenText"] if result['matches'] else styles["RedText"]
        ))
        
        current_row.append(cell_content)
        
        if len(current_row) == 6:
            table_data.append(current_row)
            current_row = []
    
    if current_row:
        while len(current_row) < 6:
            current_row.append("")
        table_data.append(current_row)
    
    if table_data:
        table = Table(table_data, colWidths=[110]*5)
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
    
    add_statistics_to_pdf(story, stats, styles)
    # generacion de pdf
    doc.build(story)
    print(f"PDF generado: {pdf_name}")
    print(f"Precisión total: {stats['accuracy']:.2f}%")

def clean_prediction(text):
    """
    Limpia la predicción eliminando caracteres no numéricos y palabras prohibidas
    
    Args:
        text (str): Texto de la predicción
        
    Returns:
        str: Predicción limpia
    """
    # Lista de palabras y patrones a eliminar
    forbidden_patterns = [
        r'Number:', r'Truck', r'Vehicle', r'ID', r'No\.?', 
        r'[A-Za-z]', r'#', r'\(.*?\)', r'\[.*?\]', r'Unit'
    ]
    cleaned = text.lower()
    
    # patrones prohibidos
    for pattern in forbidden_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'[^\d\s]', '', cleaned)
    cleaned = ' '.join(cleaned.split())
    
    numbers = cleaned.split()
    if numbers:
        cleaned = max(numbers, key=len)
    
    return cleaned

def process_model_output(output, prompt_length, processor):
    """
    Procesa la salida del modelo aplicando post-procesamiento
    
    Args:
        output: Salida del modelo
        prompt_length: Longitud del prompt
        processor: Procesador del modelo
        
    Returns:
        str: Predicción procesada
    """
    generation = output[0][prompt_length:]
    result = processor.decode(generation, skip_special_tokens=True)
    result = result.replace("User:", "").replace("Assistant:", "").strip()
    
    cleaned_result = clean_prediction(result)
    
    return cleaned_result

if __name__ == "__main__":
  
    generate_pdf()
