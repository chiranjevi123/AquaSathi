# AquaSathi  💧🌿

**AI-Powered Water Quality Advisor for Vulnerable Communities**

AquaSathi is an accessible, lightweight solution designed to help elderly and specially-abled individuals assess water quality at home or in community settings using AI, ML, and a simple dashboard interface.

---

## 🌟 Features

- ✅ **AI-Powered Water Quality Assessment:** Uses a simple ML model to classify water as `Safe` or `Unsafe`.
- ✅ **Crowdsourced Water Safety Map:** A new feature that allows users to contribute data to a global map, highlighting safe and unsafe water zones in their community.
- ✅ **Voice Command Feature:** Users can interact with the app using simple voice commands (e.g., "Check Water"), making it more accessible.
- ✅ **Inclusive UI Mode:** Provides an accessibility toggle for larger fonts and high-contrast themes for users with visual impairments.
- ✅ **Multilingual Support:** The dashboard offers in-browser translation to local languages like Hindi, catering to a wider audience.
- ✅ **Educational & Gamified Mode:** A section dedicated to educating users about water quality parameters in an easy-to-understand format.
- ✅ Simple Flask Web Dashboard
- ✅ Ready for IoT integration (future scope)

---

## 🧠 Machine Learning (ml-model)

- Uses logistic regression to classify water quality.
- Input: Color, turbidity, pH values
- Output: `Safe` or `Unsafe`
- Dataset: Simulated `water_quality.csv` (included)

### Run Model Training
```bash
cd ml-model
python train_model.py
