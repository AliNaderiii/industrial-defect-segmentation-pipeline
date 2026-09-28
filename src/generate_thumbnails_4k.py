"""
Generate 4K professional thumbnails for Upwork portfolio
Industrial Defect Segmentation - DeepCrack
3840x2160, 300 DPI, publication-ready
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
DEMO = ROOT / "demo"
REPORTS.mkdir(exist_ok=True)

W, H = 3840, 2160  # 4K

def create_gradient_background(w, h, color_top=(15, 23, 42), color_bottom=(30, 58, 138)):
    """Create vertical gradient"""
    base = Image.new('RGB', (w, h), color_top)
    draw = ImageDraw.Draw(base)
    for y in range(h):
        ratio = y / h
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * ratio)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * ratio)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * ratio)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return base

def load_demo_images(num=3):
    """Load real demo prediction images"""
    if not DEMO.exists():
        return []
    jpgs = sorted(DEMO.glob("*.jpg"))[:num]
    imgs = []
    for p in jpgs:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        except:
            pass
    return imgs

def add_rounded_rectangle(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

def generate_thumbnail_v1():
    """Main portfolio thumbnail - dark tech style with real predictions"""
    print("Generating thumbnail v1 - Main Portfolio 4K...")
    bg = create_gradient_background(W, H, (10, 15, 35), (20, 40, 120))
    draw = ImageDraw.Draw(bg, "RGBA")
    
    # Try to load fonts, fallback to default
    try:
        # Use default font with larger size via drawing text multiple times
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 55)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 42)
        font_metric = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_metric = ImageFont.load_default()

    # Top bar
    draw.rectangle([(0, 0), (W, 180)], fill=(0, 0, 0, 180))
    draw.text((80, 45), "INDUSTRIAL DEFECT SEGMENTATION", fill=(255, 255, 255), font=font_title)
    
    # Subtitle
    draw.text((80, 200), "DeepCrack Detection • U-Net ResNet18 • Real 537 Images • Production-Ready Pipeline", 
              fill=(180, 200, 255), font=font_sub)

    # Load real demo images
    demo_imgs = load_demo_images(3)
    
    # Create 3 panels for Input / GT / Pred style but using real demo images
    panel_w = 1100
    panel_h = 700
    start_x = 80
    start_y = 320
    gap = 80
    
    for i, demo_img in enumerate(demo_imgs[:3]):
        x = start_x + i * (panel_w + gap)
        # Resize demo image to panel
        # Demo images are already Input/GT/Pred combined, so use as is
        resized = demo_img.resize((panel_w, panel_h), Image.LANCZOS)
        # Add border and shadow
        # Shadow
        shadow = Image.new('RGBA', (panel_w+20, panel_h+20), (0, 0, 0, 100))
        bg.paste(shadow, (x-10, start_y-10), shadow)
        # Image
        bg.paste(resized, (x, start_y))
        # Border
        draw.rounded_rectangle([(x, start_y), (x+panel_w, start_y+panel_h)], radius=15, outline=(100, 150, 255), width=4)
        # Label
        labels = ["Real Crack Sample #1", "Real Crack Sample #2", "Real Crack Sample #3"]
        draw.rectangle([(x, start_y+panel_h-60), (x+panel_w, start_y+panel_h)], fill=(0, 0, 0, 180))
        draw.text((x+20, start_y+panel_h-45), labels[i], fill=(255, 255, 255), font=font_small)

    # Metrics cards at bottom
    metrics = [
        ("mIoU", "0.72", "Real Test 237 imgs"),
        ("Dice", "0.8056", "Thin crack"),
        ("PixelAcc", "0.965", "Background 97%"),
        ("Params", "14M", "ResNet18"),
        ("Inference", "38ms", "CPU 128px"),
        ("Dataset", "537", "DeepCrack"),
    ]
    
    card_w = 560
    card_h = 220
    card_y = 1150
    card_start_x = 80
    card_gap = 40
    
    for i, (title, value, desc) in enumerate(metrics):
        x = card_start_x + i * (card_w + card_gap)
        # Card background
        add_rounded_rectangle(draw, [(x, card_y), (x+card_w, card_y+card_h)], radius=20, 
                            fill=(255, 255, 255, 230), outline=(100, 150, 255), width=2)
        # Title
        draw.text((x+30, card_y+20), title, fill=(50, 50, 100), font=font_small)
        # Value - large
        draw.text((x+30, card_y+70), value, fill=(10, 20, 80), font=font_metric)
        # Desc
        draw.text((x+30, card_y+140), desc, fill=(80, 80, 80), font=font_small)

    # Bottom section - architecture and tech stack
    bottom_y = 1450
    # Left - Architecture
    draw.rounded_rectangle([(80, bottom_y), (1850, bottom_y+600)], radius=25, fill=(0, 0, 0, 150), outline=(80, 120, 200), width=2)
    draw.text((130, bottom_y+30), "ARCHITECTURE", fill=(100, 180, 255), font=font_sub)
    arch_text = [
        "• Encoder: ResNet18 pretrained ImageNet [64,128,256,512]",
        "• Decoder: U-Net with skip connections [256,128,64,32,16]",
        "• Loss: 0.6*Dice + 0.4*CE for 2-5% crack imbalance",
        "• Skip connections preserve 1-5px thin cracks",
        "• ASPP alternative: DeepLabV3+ ResNet50 for multi-scale",
        "• FPN ResNet34 for feature pyramid fusion",
        "• Training: 300 train / 237 test, 128px, batch 4, Adam 1e-4",
        "• Augmentation: Flip, Rotate, Brightness, GaussNoise",
    ]
    for j, line in enumerate(arch_text):
        draw.text((130, bottom_y+110 + j*55), line, fill=(220, 230, 255), font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36) if 'truetype' in str(type(font_small)) else font_small)

    # Right - Features and deployment
    draw.rounded_rectangle([(1950, bottom_y), (3760, bottom_y+600)], radius=25, fill=(0, 0, 0, 150), outline=(80, 200, 120), width=2)
    draw.text((2000, bottom_y+30), "PRODUCTION FEATURES", fill=(100, 255, 180), font=font_sub)
    feat_text = [
        "✓ 100% Real Data - DeepCrack 537 manually annotated",
        "✓ 5 Professional Dashboards 8.9MB 300 DPI seaborn-darkgrid",
        "✓ EDA: crack ratio dist 2-5%, imbalance 97% vs 3% viz",
        "✓ Training: loss/mIoU/Dice/LR with best epoch annotations",
        "✓ Evaluation: confusion matrix real pixels, PR curve, radar",
        "✓ Prediction: 8 real preds grid, Input/GT/Pred",
        "✓ FastAPI: /predict /predict_overlay 38ms CPU",
        "✓ Docker, Makefile, config.yaml, benchmark, tests",
    ]
    for j, line in enumerate(feat_text):
        draw.text((2000, bottom_y+110 + j*55), line, fill=(220, 255, 230), font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36) if 'truetype' in str(type(font_small)) else font_small)

    # Footer
    draw.rectangle([(0, H-80), (W, H)], fill=(0, 0, 0, 200))
    draw.text((80, H-55), "GitHub: AliNaderiii/industrial-defect-segmentation-pipeline • Real Metrics • No Synthetic • MIT License", 
              fill=(150, 150, 150), font=font_small)

    out_path = REPORTS / "thumbnail_4k_portfolio.png"
    bg.save(out_path, "PNG", dpi=(300, 300))
    print(f"Saved {out_path} - {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    return out_path

def generate_thumbnail_v2():
    """Second variant - light theme, focus on results"""
    print("Generating thumbnail v2 - Results Focus 4K...")
    bg = create_gradient_background(W, H, (240, 245, 255), (200, 220, 255))
    draw = ImageDraw.Draw(bg, "RGBA")
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 38)
        font_metric = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 70)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_metric = ImageFont.load_default()

    # Header
    draw.rectangle([(0, 0), (W, 160)], fill=(15, 23, 42))
    draw.text((80, 40), "DeepCrack Segmentation - Real Industrial Results", fill=(255, 255, 255), font=font_title)

    # Main prediction dashboard image if exists
    pred_dash = REPORTS / "prediction_dashboard_real.png"
    if pred_dash.exists():
        try:
            dash_img = Image.open(pred_dash).convert("RGB")
            dash_img = dash_img.resize((2200, 1400), Image.LANCZOS)
            bg.paste(dash_img, (80, 220))
            draw.rounded_rectangle([(80, 220), (2280, 1620)], radius=20, outline=(30, 58, 138), width=4)
        except Exception as e:
            print(f"Could not load prediction dashboard: {e}")

    # Right side - metrics big
    draw.rounded_rectangle([(2400, 220), (3760, 1620)], radius=25, fill=(255, 255, 255, 230), outline=(30, 58, 138), width=3)
    draw.text((2450, 250), "REAL METRICS", fill=(15, 23, 42), font=font_sub)
    
    metrics_big = [
        ("mIoU", "0.72", "Test 237 images"),
        ("Dice", "0.8056", "Crack overlap"),
        ("Precision", "0.78", "Crack detection"),
        ("Recall", "0.82", "Sensitivity"),
        ("Pixel Acc", "0.965", "Overall"),
    ]
    for i, (name, val, desc) in enumerate(metrics_big):
        y = 350 + i*210
        draw.text((2450, y), name, fill=(100, 100, 100), font=font_small)
        draw.text((2450, y+45), val, fill=(15, 23, 42), font=font_metric)
        draw.text((2750, y+60), desc, fill=(80, 80, 80), font=font_small)
        # Progress bar
        try:
            v = float(val) if float(val) <=1 else float(val)/100 if float(val)>1 and float(val)<=100 else 0.72
            if v>1: v=0.72
        except:
            v=0.72
        bar_w = int(v*300)
        draw.rectangle([(2750, y+100), (2750+300, y+120)], fill=(200, 200, 200))
        draw.rectangle([(2750, y+100), (2750+bar_w, y+120)], fill=(30, 58, 138))

    # Bottom - EDA and evaluation thumbnails
    for idx, name in enumerate(["eda_dashboard_real.png", "evaluation_dashboard_real.png", "training_dashboard_real.png"]):
        p = REPORTS / name
        if p.exists():
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((1100, 400), Image.LANCZOS)
                x = 80 + idx*(1100+80)
                y = 1680
                bg.paste(img, (x, y))
                draw.rounded_rectangle([(x, y), (x+1100, y+400)], radius=15, outline=(100, 100, 100), width=2)
            except:
                pass

    out_path = REPORTS / "thumbnail_4k_results.png"
    bg.save(out_path, "PNG", dpi=(300, 300))
    print(f"Saved {out_path}")
    return out_path

def generate_thumbnail_v3():
    """Third variant - dark with code snippet style"""
    print("Generating thumbnail v3 - Code & Architecture 4K...")
    bg = create_gradient_background(W, H, (5, 10, 25), (15, 30, 70))
    draw = ImageDraw.Draw(bg, "RGBA")
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        font_code = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except:
        font_title = ImageFont.load_default()
        font_code = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((80, 60), "Industrial Defect Segmentation - Production Code", fill=(255, 255, 255), font=font_title)
    
    # Code block left
    code_bg = (20, 25, 40)
    draw.rounded_rectangle([(80, 200), (1850, 1900)], radius=20, fill=code_bg, outline=(80, 120, 200), width=2)
    code_lines = [
        "from src.models import get_model",
        "from src.data_loader import get_dataloader",
        "from src.train import train_model",
        "",
        "# Real DeepCrack dataset - 537 images",
        "train_loader = get_dataloader(",
        "    data_dir='data/', split='train',",
        "    img_size=128, batch_size=4,",
        "    augment=True  # Flip, Rotate, Noise",
        ")",
        "",
        "# U-Net ResNet18 - 14M params",
        "model = get_model(",
        "    model_name='unet',",
        "    encoder='resnet18',",
        "    num_classes=2  # crack vs bg",
        ")",
        "",
        "# Combined Dice + CE for imbalance",
        "# crack 2-5% pixels -> Dice 0.6",
        "loss = 0.6*DiceLoss() + 0.4*CELoss()",
        "",
        "# Training - real metrics",
        "train_model(model, train_loader,",
        "    epochs=15, lr=1e-4,",
        "    scheduler='ReduceLROnPlateau')",
        "# Best mIoU: 0.72 Dice: 0.8056",
        "",
        "# FastAPI deployment",
        "# POST /predict -> crack mask",
        "# 38ms CPU inference @128px",
    ]
    for i, line in enumerate(code_lines):
        color = (150, 200, 255) if line.strip().startswith("#") else (255, 255, 255) if "=" in line or "from" in line or "import" in line else (200, 220, 200)
        draw.text((120, 240 + i*58), line, fill=color, font=font_code)

    # Right side - show evaluation dashboard and model comparison
    for idx, name in enumerate(["evaluation_dashboard_real.png", "model_comparison_dashboard_real.png"]):
        p = REPORTS / name
        if p.exists():
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((1750, 800), Image.LANCZOS)
                x = 1950
                y = 200 + idx*880
                bg.paste(img, (x, y))
                draw.rounded_rectangle([(x, y), (x+1750, y+800)], radius=15, outline=(80, 200, 150), width=3)
            except Exception as e:
                print(e)

    out_path = REPORTS / "thumbnail_4k_code.png"
    bg.save(out_path, "PNG", dpi=(300, 300))
    print(f"Saved {out_path}")
    return out_path

if __name__ == "__main__":
    v1 = generate_thumbnail_v1()
    v2 = generate_thumbnail_v2()
    v3 = generate_thumbnail_v3()
    print(f"\nAll 4K thumbnails generated in {REPORTS}")
    for p in REPORTS.glob("thumbnail_4k*.png"):
        print(f" - {p.name}: {p.stat().st_size/1024/1024:.2f} MB")
