"use client";

import { useState, useRef, useEffect } from "react";
import { useLanguage } from "@/components/LanguageContext";
import { useAuth, ProtectedRoute } from "@/components/AuthContext";

interface ChatMsg {
  text: string;
  sender: "user" | "bot";
}

function ChatbotContent() {
  const { t, lang } = useLanguage();
  const { token } = useAuth();
  const [mode, setMode] = useState<"none" | "text" | "voice">("none");
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [isListening, setIsListening] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    setMessages([{ text: t("botWelcome"), sender: "bot" }]);
  }, [t]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (msgText: string) => {
    if (!msgText.trim()) return;

    setMessages(prev => [...prev, { text: msgText, sender: "user" }]);
    setInput("");

    try {
      const response = await fetch("http://localhost:5005/webhooks/rest/webhook", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({ sender: "user", message: msgText })
      });

      if (response.status === 401) {
        setMessages(prev => [...prev, { text: "⚠️ Session expired. Please login again.", sender: "bot" }]);
        return;
      }

      const data = await response.json();
      data.forEach((botMsg: any) => {
        if (botMsg.text) {
          setMessages(prev => [...prev, { text: botMsg.text, sender: "bot" }]);
        }
      });
    } catch (err) {
      console.error("Error connecting to server:", err);
      setMessages(prev => [...prev, { text: "⚠️ Error: Could not connect to server.", sender: "bot" }]);
    }
  };

  const handleSend = () => {
    sendMessage(input);
  };

  const startVoiceInput = () => {
    if (!("webkitSpeechRecognition" in window)) {
      alert("❌ Voice recognition not supported in this browser.");
      return;
    }

    const SpeechRecognition = (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognitionRef.current = recognition;

    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = lang === "hi" ? "hi-IN" : lang === "pa" ? "pa-IN" : "en-US";

    recognition.onstart = () => {
      setIsListening(true);
      setMessages(prev => [...prev, { text: "🎤 Listening... please start speaking.", sender: "bot" }]);
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      sendMessage(transcript);
      stopVoiceInput(false);
    };

    recognition.onerror = (event: any) => {
      setMessages(prev => [...prev, { text: "⚠️ Voice input error: " + event.error, sender: "bot" }]);
      stopVoiceInput();
    };

    recognition.onend = () => {
      if (recognitionRef.current) stopVoiceInput();
    };

    recognition.start();
  };

  const stopVoiceInput = (showMessage = true) => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
      setIsListening(false);
      if (showMessage) {
        setMessages(prev => [...prev, { text: "🛑 Stopped listening.", sender: "bot" }]);
      }
    }
  };

  if (mode === "none") {
    return (
      <section className="flex flex-col items-center justify-center min-h-[80vh] bg-gradient-to-r from-green-50 to-yellow-50">
        <h1 className="text-3xl font-bold text-green-700 mb-6">{t("chooseInput")}</h1>
        <div className="flex gap-6">
          <button onClick={() => setMode("text")} className="px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition">
            Text Mode
          </button>
          <button onClick={() => setMode("voice")} className="px-6 py-3 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition">
            Voice Mode
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="flex items-center justify-center min-h-screen bg-gradient-to-br from-green-50 to-yellow-50 pt-6">
      <div className="chat-container w-full max-w-md h-[80vh] bg-white/90 backdrop-blur-md rounded-2xl shadow-2xl flex flex-col">
        {/* Header */}
        <div className="text-xl font-semibold text-green-700 p-4 border-b border-gray-200 rounded-t-2xl bg-green-50 flex justify-between items-center">
          <span>{t("chatHeader")}</span>
          <button onClick={() => setMode("none")} className="text-sm text-gray-500 hover:text-gray-700">Back</button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`message p-3 rounded-lg shadow-sm w-fit max-w-[75%] fade-in ${msg.sender === 'bot' ? 'bg-gray-100' : 'bg-green-100 ml-auto'}`}
            >
              {msg.text}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="flex items-center gap-3 p-4 border-t border-gray-200 bg-gradient-to-r from-white to-green-50 rounded-b-2xl">
          {mode === "text" ? (
            <>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                className="flex-1 rounded-lg p-3 resize-none focus:outline-none focus:ring-2 focus:ring-green-400 bg-white border border-gray-300"
                rows={2}
                placeholder="Type your message..."
              />
              <button onClick={handleSend} className="px-5 py-3 bg-green-500 text-white font-medium rounded-lg shadow hover:bg-green-600 transition">
                ➤
              </button>
            </>
          ) : (
            <div className="flex-1 flex justify-center">
              {!isListening ? (
                <button onClick={startVoiceInput} className="px-6 py-3 bg-green-100 text-green-700 rounded-full shadow hover:bg-green-200 transition text-lg flex items-center gap-2">
                  🎤 Tap to Speak
                </button>
              ) : (
                <button onClick={() => stopVoiceInput()} className="px-6 py-3 bg-red-100 text-red-600 rounded-full shadow hover:bg-red-200 transition text-lg flex items-center gap-2">
                  ⏹️ Stop Listening
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export default function Chatbot() {
  return (
    <ProtectedRoute>
      <ChatbotContent />
    </ProtectedRoute>
  );
}
