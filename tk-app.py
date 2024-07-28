import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from super_image import EdsrModel, ImageLoader
from diffusers import StableDiffusionPipeline
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class MobileApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mobile App")
        self.root.geometry("360x640")
        self.frame = tk.Frame(root)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, bg="white", width=400, height=640)

        # Viewing image
        self.back_icon = tk.PhotoImage(file="img/back5.png")
        self.canvas.create_image(10, 15, anchor=tk.NW, image=self.back_icon)
        self.canvas.create_text(50, 14, anchor=tk.NW, text="Tambah Hidangan Baru", fill="black", font=('Helvetica 17 bold'))
        self.question_info = tk.PhotoImage(file="img/quest2.png")
        self.canvas.create_image(320, 15, anchor=tk.NW, image=self.question_info)

        # Model text generation
        model_name = "t5-base"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def generate_content(prompt, max_length=100):
            inputs = tokenizer.encode_plus(prompt, add_special_tokens=True, max_length=max_length, return_attention_mask=True, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        def open_image():
            def on_back():
                popup.destroy()

            def on_yes(filepath):
                image = Image.open(filepath)
                model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=3)
                inputs = ImageLoader.load_image(image)
                preds = model(inputs)
                preds.thumbnail((110, 110))
                photo = ImageTk.PhotoImage(preds)
                image_label = tk.Label(self.frame, image=photo)
                image_label.image = photo
                image_label.place(x=10, y=80, width=110, height=110)

                # Resize and crop logic (needs proper implementation)
                # Example: preds = preds.resize((new_width, new_height))
                preds = preds.crop((0, 0, new_width, crop_height))

                # Generative (ensure torch, CUDA are configured correctly)
                pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to("cuda")
                prompt = "A photorealistic depiction..."
                generated_image = pipe(prompt, guidance_scale=7.5, num_inference_steps=20).images[0]
                generated_image = generated_image.resize((new_width, crop_height))
                preds.paste(generated_image, (0, 0))

                popup.destroy()

            def show_popup(file_path):
                global popup
                popup = tk.Toplevel(root)
                popup.title("Konfirmasi")
                popup.geometry("300x200")
                label = tk.Label(popup, text="Tingkatkan kualitas Foto", font=('Helvetica', 14))
                label.pack(pady=20)
                ok_button = tk.Button(popup, text="Ya", command=lambda: on_yes(file_path))
                ok_button.pack(side=tk.LEFT, padx=20, pady=10)
                cancel_button = tk.Button(popup, text="Tidak", command=on_back)
                cancel_button.pack(side=tk.RIGHT, padx=20, pady=10)

            file_path = filedialog.askopenfilename(initialdir="/", title="Select an image", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
            if file_path:
                show_popup(file_path)

        self.question_info = tk.PhotoImage(file="img/picture.png")
        self.image_input = tk.Button(self.frame, image=self.question_info, command=open_image)
        self.image_input.place(x=10, y=80, width=110, height=110)
        self.canvas.create_text(125, 87, anchor=tk.NW, text="Maks. 4 foto, masing-masing hingga 2", fill="grey", font=('Helvetica 12'))
        self.canvas.create_text(125, 100, anchor=tk.NW, text="MB.", fill="grey", font=('Helvetica 12'))
        self.canvas.pack()

        # Nama Hidangan
        self.canvas.create_text(10, 210, anchor=tk.NW, text="Nama hidangan*", fill="grey", font=('Helvetica 12'))
        self.product_name = tk.Entry(self.frame, width=35, fg='grey', bg='#ffffff', bd=1, highlightthickness=0, highlightcolor='blue')
        self.product_name.insert(10, 'Beri nama hidangan ini')
        self.product_name.place(x=10, y=230)

        def on_entry_click(event):
            if self.product_name.get() == 'Beri nama hidangan ini':
                self.product_name.delete(0, tk.END)
                self.product_name.config(fg='black')

        def on_focus_out(event):
            if self.product_name.get() == '':
                self.product_name.insert(0, 'Beri nama hidangan ini')
                self.product_name.config(fg='grey')

        self.product_name.bind('<FocusIn>', on_entry_click)
        self.product_name.bind('<FocusOut>', on_focus_out)

        # Deskripsi produk
        self.canvas.create_text(10, 280, anchor=tk.NW, text="Deskripsi", fill="grey", font=('Helvetica 12'))
        self.description = tk.Entry(self.frame, width=35, fg='grey', bg='#ffffff', bd=1, highlightthickness=0, highlightcolor='blue')
        self.description.insert(15, 'Bahan-bahan, cara pembuatan, dll.')
        self.description.place(x=10, y=300)

        def on_entry_click_desc(event):
            if self.description.get() == 'Bahan-bahan, cara pembuatan, dll.':
                self.description.delete(0, tk.END)
                self.description.config(fg='black')

        def on_focus_out_desc(event):
            if self.description.get() == '':
                self.description.insert(0, 'Bahan-bahan, cara pembuatan, dll.')
                self.description.config(fg='grey')

        self.description.bind('<FocusIn>', on_entry_click_desc)
        self.description.bind('<FocusOut>', on_focus_out_desc)

        # Button generative text
        def show_input():
            prompt = self.description.get()
            generated_content = generate_content(prompt)
            self.description.delete(0, tk.END)
            self.description.insert(0, generated_content)

        self.button = tk.Button(self.frame, text="Buat otomatis", command=show_input, width=20, fg="grey", font=('Helvetica 12'))
        self.button.place(x=230, y=295)

if __name__ == "__main__":
    root = tk.Tk()
    app = MobileApp(root)
    root.mainloop()