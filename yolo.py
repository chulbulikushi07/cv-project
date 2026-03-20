from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("https://media.discordapp.net/attachments/1136005468792295546/1477725005075841094/IMG_0866.png?ex=69a5ce28&is=69a47ca8&hm=218be7a63325fb167c5b26975799e38c2a2c3a2bf926be084ce51f94821ba480&=&format=webp&quality=lossless&width=203&height=438")

results[0].show()