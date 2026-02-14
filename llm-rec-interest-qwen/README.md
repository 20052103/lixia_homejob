(llmrec) PS D:\repo\lixia_homejob\llm-rec-interest-qwen> python "D:\repo\lixia_homejob\llm-rec-interest-qwen\scripts\07_final_compare_lora_on_off.py"
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████| 339/339 [00:03<00:00, 96.74it/s, Materializing param=model.norm.weight]
# LoRA OFF vs LoRA ON (Qwen2.5-7B-Instruct)

## Case: Vehicles + anime + fishing

### Prompt

```text
<|system|>
You are a helpful assistant.
<|user|>
Here are recent watched video topics (labels). Summarize the user's interests.
1) Vehicle, Car, Renault, Ford Focus
2) Vehicle, Aircraft, Airplane, Airline
3) Cartoon, Inuyasha
4) Fishing, Recreational fishing
<|assistant|>

```

The following generation flags are not valid and may be ignored: ['top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
### LoRA OFF (base behavior)

```text
The user shows interest in vehicles, including cars and aircraft, as well as cartoons. They also have an interest in recreational activities such as fishing. Specifically, they seem to be interested in specific car brands like Renault and Ford Focus, and are fans of the cartoon Inuyasha. They also show interest in airplanes and airlines. Overall, their interests span automotive, aviation, entertainment, and outdoor activities.
```

### LoRA ON

```text
Primary interests: Car, Renault, Ford Focus.
Secondary interests: Aircraft, Airplane, Airline, Inuyasha.
Recommendation direction: more content related to Car and Renault; occasional videos about Aircraft and Airplane.
```


---

## Case: Sports + music + cooking

### Prompt

```text
<|system|>
You are a helpful assistant.
<|user|>
User recently watched videos with these topics. What are their interests?
1) Basketball, NBA, Golden State Warriors
2) Concert, Musician, Drum, Drummer
3) Game, Association football, Premier League
4) Cooking, Recipe, Vegetable
<|assistant|>

```

### LoRA OFF (base behavior)

```text
Based on the topics of the videos the user recently watched, we can infer that their interests include:

1. **Sports**:
   - Basketball and specifically the NBA and Golden State Warriors indicate an interest in professional basketball.
   - Association football (soccer) and the Premier League suggest an interest in international football.

2. **Music**:
   - Concerts and musicians, particularly those involving drums or drummers, point to an interest in live music performances and possibly certain genres of music that feature prominent drumming.

3. **Cooking and Food**:
   - Cooking and recipes, especially those involving vegetables, indicate an interest in culinary arts and healthy eating.

In summary, the user appears
```

### LoRA ON

```text
Primary interests: Basketball, NBA, Golden State Warriors.
Secondary interests: Musician, Drum, Drummer, Association football.
Recommendation direction: more content related to Basketball and NBA; occasional videos about Musician and Drum.
```


---

## Case: Craft + toys + running + dance

### Prompt

```text
<|system|>
You are a helpful assistant.
<|user|>
Recent watched topics:
1) Origami, Felt
2) Toy, Lego, Kinder Surprise, Playground
3) Running, Marathon
4) Dance, Performance art
Summarize the user's interests.
<|assistant|>

```

### LoRA OFF (base behavior)

```text
Based on the recent watched topics, it appears that the user has interests in crafting and DIY projects (Origami and Felt), toys and play (Lego, Kinder Surprise, Playground), physical activities and sports (Running, Marathon), and performing arts (Dance, Performance art). These interests suggest a diverse range of hobbies and activities that include creativity, physical fitness, and artistic expression.
```

### LoRA ON

```text
Primary interests: Origami, Felt, Lego.
Secondary interests: Kinder Surprise, Playground, Running, Marathon.
Recommendation direction: more content related to Origami and Felt; occasional posts about Kinder Surprise and Playground. To occasional mentions of Running and Marathon.
```


---

## Case: Beauty + fashion

### Prompt

```text
<|system|>
You are a helpful assistant.
<|user|>
Topics watched:
1) Makeup, Cosmetics, Lipstick
2) Hairstyle, Wig
3) Fashion, Streetwear
4) Skincare
Summarize the user's interests.
<|assistant|>

```

### LoRA OFF (base behavior)

```text
The user has shown interest in beauty and fashion-related topics, specifically focusing on makeup and skincare, hairstyles and wigs, and fashion trends including streetwear.
```

### LoRA ON

```text
Primary interests: Makeup, Cosmetics, Lipstick.
Secondary interests: Hairstyle, Wig, Fashion, Streetwear.
Recommendation direction: more content related to Makeup and Cosmetics; occasional videos about Hairstyle and Wig. To maintain engagement, occasional videos