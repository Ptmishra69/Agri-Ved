"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";

type Language = "en" | "hi" | "pa";

interface LanguageContextType {
  lang: Language;
  setLang: (lang: Language) => void;
  t: (key: string) => string;
}

const translations: Record<Language, Record<string, string>> = {
  en: {
    home: "Home",
    chatbot: "Chatbot",
    weather: "Weather Report",
    market: "Market Price",
    services: "Services",
    contact: "Contact Us",
    login: "Login",
    signup: "Sign Up",
    heroTitle: "Great Farming starts with Agri-Ved",
    heroDesc: "We help farmers with soil recommendations, smart farming tools, and AI-powered chatbot support.",
    getStarted: "Let’s Get Started!",
    servicesTitle: "Services we offer",
    servicesCrop: "Crop Recommendations",
    servicesCropDesc: "AI-driven system to help farmers choose the most suitable crop by analyzing soil & environment.",
    servicesChatbot: "ChatBot",
    servicesChatbotDesc: "Smart, AI-powered assistant providing instant responses, real-time support, and guidance 24/7.",
    servicesWeather: "Weather Prediction",
    servicesWeatherDesc: "Helps farmers make smarter decisions on sowing, irrigation, and harvesting to reduce crop losses.",
    servicesFertilizer: "Fertilizer Amount",
    servicesFertilizerDesc: "Ensures correct fertilizer usage, improving crop growth while protecting soil health.",
    contactTitle: "Contact Us",
    emailLabel: "Email:",
    phoneLabel: "Phone:",
    addressLabel: "Address:",
    address: "Greater Noida Institute of Technology, UP, India",
    chooseInput: "Choose Input Method",
    chatHeader: "Agri-Ved ChatBot 🤖",
    botWelcome: "Hello! 👋 I am Agri-Ved Bot. How can I help you today?"
  },
  hi: {
    home: "होम",
    chatbot: "चैटबॉट",
    weather: "मौसम रिपोर्ट",
    market: "बाजार मूल्य",
    services: "सेवाएँ",
    contact: "संपर्क करें",
    login: "लॉगिन",
    signup: "साइन अप",
    heroTitle: "महान खेती की शुरुआत एग्री-वेड़ से होती है",
    heroDesc: "हम किसानों को मिट्टी की सिफारिशें, स्मार्ट खेती उपकरण और एआई चैटबॉट सहायता प्रदान करते हैं।",
    getStarted: "शुरू करें!",
    servicesTitle: "हमारी सेवाएँ",
    servicesCrop: "फसल अनुशंसाएँ",
    servicesCropDesc: "एआई आधारित प्रणाली जो मिट्टी और पर्यावरण का विश्लेषण कर किसानों को सबसे उपयुक्त फसल चुनने में मदद करती है।",
    servicesChatbot: "चैटबॉट",
    servicesChatbotDesc: "स्मार्ट, एआई-संचालित सहायक जो 24/7 तुरंत उत्तर और मार्गदर्शन प्रदान करता है।",
    servicesWeather: "मौसम पूर्वानुमान",
    servicesWeatherDesc: "किसानों को बुवाई, सिंचाई और कटाई पर समझदारीपूर्ण निर्णय लेने में मदद करता है।",
    servicesFertilizer: "उर्वरक मात्रा",
    servicesFertilizerDesc: "सही उर्वरक उपयोग सुनिश्चित करता है, जिससे फसल बेहतर होती है और मिट्टी की सेहत सुरक्षित रहती है।",
    contactTitle: "संपर्क करें",
    emailLabel: "ईमेल:",
    phoneLabel: "फ़ोन:",
    addressLabel: "पता:",
    address: "ग्रेटर नोएडा इंस्टिट्यूट ऑफ टेक्नोलॉजी, यूपी, भारत",
    chooseInput: "इनपुट विधि चुनें",
    chatHeader: "एग्री-वेड़ चैटबॉट 🤖",
    botWelcome: "नमस्ते! 👋 मैं एग्री-वेड़ बॉट हूँ। मैं आपकी कैसे मदद कर सकता हूँ?"
  },
  pa: {
    home: "ਘਰ",
    chatbot: "ਚੈਟਬੋਟ",
    weather: "ਮੌਸਮ ਰਿਪੋਰਟ",
    market: "ਬਾਜ਼ਾਰ ਕੀਮਤ",
    services: "ਸੇਵਾਵਾਂ",
    contact: "ਸਾਡੇ ਨਾਲ ਸੰਪਰਕ ਕਰੋ",
    login: "ਲਾਗਇਨ",
    signup: "ਸਾਈਨ ਅੱਪ",
    heroTitle: "ਵਧੀਆ ਖੇਤੀਬਾੜੀ ਦੀ ਸ਼ੁਰੂਆਤ Agri-Ved ਨਾਲ ਹੁੰਦੀ ਹੈ",
    heroDesc: "ਅਸੀਂ ਕਿਸਾਨਾਂ ਨੂੰ ਮਿੱਟੀ ਦੀ ਸਿਫ਼ਾਰਸ਼ਾਂ, ਸਮਾਰਟ ਖੇਤੀ ਦੇ ਸੰਦ ਅਤੇ AI ਚੈਟਬੋਟ ਸਹਾਇਤਾ ਦਿੰਦੇ ਹਾਂ।",
    getStarted: "ਸ਼ੁਰੂ ਕਰੋ!",
    servicesTitle: "ਅਸੀਂ ਦਿੱਤੀਆਂ ਸੇਵਾਵਾਂ",
    servicesCrop: "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼",
    servicesCropDesc: "AI ਪ੍ਰਣਾਲੀ ਜੋ ਮਿੱਟੀ ਅਤੇ ਵਾਤਾਵਰਣ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਕੇ ਕਿਸਾਨਾਂ ਨੂੰ ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲ ਚੁਣਨ ਵਿੱਚ ਮਦਦ ਕਰਦੀ ਹੈ।",
    servicesChatbot: "ਚੈਟਬੋਟ",
    servicesChatbotDesc: "ਸਮਾਰਟ AI ਸਹਾਇਕ ਜੋ 24/7 ਤੁਰੰਤ ਜਵਾਬ ਅਤੇ ਮਾਰਗਦਰਸ਼ਨ ਪ੍ਰਦਾਨ ਕਰਦਾ ਹੈ।",
    servicesWeather: "ਮੌਸਮ ਦੀ ਭਵਿੱਖਬਾਣੀ",
    servicesWeatherDesc: "ਕਿਸਾਨਾਂ ਨੂੰ ਬੀਜਾਈ, ਸਿੰਚਾਈ ਅਤੇ ਕੱਟਾਈ ਬਾਰੇ ਸਮਝਦਾਰ ਫੈਸਲੇ ਕਰਨ ਵਿੱਚ ਮਦਦ ਕਰਦਾ ਹੈ।",
    servicesFertilizer: "ਖਾਦ ਦੀ ਮਾਤਰਾ",
    servicesFertilizerDesc: "ਸਹੀ ਖਾਦ ਦੇ ਇਸਤੇਮਾਲ ਨੂੰ ਯਕੀਨੀ ਬਣਾਉਂਦਾ ਹੈ, ਜਿਸ ਨਾਲ ਫਸਲ ਦੀ ਵਾਧਾ ਹੁੰਦੀ ਹੈ ਅਤੇ ਮਿੱਟੀ ਦੀ ਸਿਹਤ ਸੁਰੱਖਿਅਤ ਰਹਿੰਦੀ ਹੈ।",
    contactTitle: "ਸਾਡੇ ਨਾਲ ਸੰਪਰਕ ਕਰੋ",
    emailLabel: "ਈਮੇਲ:",
    phoneLabel: "ਫੋਨ:",
    addressLabel: "ਪਤਾ:",
    address: "ਗ੍ਰੇਟਰ ਨੋਇਡਾ ਇੰਸਟੀਚਿਊਟ ਆਫ ਟੈਕਨੋਲੋਜੀ, ਯੂਪੀ, ਭਾਰਤ",
    chooseInput: "ਇਨਪੁੱਟ ਢੰਗ ਚੁਣੋ",
    chatHeader: "ਐਗਰੀ-ਵੇਦ ਚੈਟਬਾਟ 🤖",
    botWelcome: "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! 👋 ਮੈਂ ਐਗਰੀ-ਵੇਦ ਬੋਟ ਹਾਂ। ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
  }
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Language>("en");

  const t = (key: string) => {
    return translations[lang][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return context;
}
