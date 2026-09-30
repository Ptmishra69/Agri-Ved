"use client";

import Image from "next/image";
import Link from "next/link";
import { useLanguage } from "@/components/LanguageContext";
import { useEffect, useRef } from "react";

export default function Home() {
  const { t } = useLanguage();
  const fadersRef = useRef<HTMLElement[]>([]);

  useEffect(() => {
    const appearOptions = {
      threshold: 0.2,
      rootMargin: "0px 0px -50px 0px"
    };

    const appearOnScroll = new IntersectionObserver(function (entries, observer) {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("show");
        observer.unobserve(entry.target);
      });
    }, appearOptions);

    fadersRef.current.forEach(fader => {
      if (fader) appearOnScroll.observe(fader);
    });

    return () => appearOnScroll.disconnect();
  }, []);

  const addToFaders = (el: HTMLElement | null) => {
    if (el && !fadersRef.current.includes(el)) {
      fadersRef.current.push(el);
    }
  };

  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section
        ref={addToFaders}
        className="relative overflow-hidden min-h-[90vh] flex items-center fade-up"
      >
        {/* Animated background blobs */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-green-50 via-yellow-50/50 to-emerald-50" />
          <div className="absolute top-20 -left-20 w-72 h-72 bg-green-200/40 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-yellow-200/30 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-emerald-100/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "2s" }} />
        </div>

        <div className="w-full px-6 md:px-16 lg:px-24 py-20">
          <div className="flex flex-col lg:flex-row items-center justify-between gap-12 lg:gap-20">
            {/* Left — Text Content */}
            <div className="max-w-xl text-center lg:text-left space-y-8">

              {/* Heading */}
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tight">
                <span className="text-gray-900">
                  {t("heroTitle").split("Agri-Ved")[0]}
                </span>
                {t("heroTitle").includes("Agri-Ved") && (
                  <span className="bg-gradient-to-r from-green-600 to-emerald-500 bg-clip-text text-transparent">
                    Agri-Ved
                  </span>
                )}
                {!t("heroTitle").includes("Agri-Ved") && (
                  <span className="bg-gradient-to-r from-green-600 to-emerald-500 bg-clip-text text-transparent">
                    {t("heroTitle")}
                  </span>
                )}
              </h1>

              {/* Description */}
              <p className="text-lg text-gray-600 leading-relaxed max-w-md mx-auto lg:mx-0">
                {t("heroDesc")}
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <Link
                  href="/chatbot"
                  className="group inline-flex items-center gap-2 px-7 py-3.5 bg-gradient-to-r from-green-500 to-emerald-600 text-white text-lg font-semibold rounded-xl shadow-lg shadow-green-200 hover:shadow-xl hover:shadow-green-300 hover:from-green-600 hover:to-emerald-700 transition-all duration-300"
                >
                  {t("getStarted")}
                  <span className="group-hover:translate-x-1 transition-transform">→</span>
                </Link>
                <Link
                  href="/#services"
                  className="inline-flex items-center gap-2 px-7 py-3.5 bg-white/70 text-green-700 text-lg font-semibold rounded-xl border border-green-200 hover:bg-green-50 hover:border-green-300 transition-all backdrop-blur-sm"
                >
                  {t("services")}
                </Link>
              </div>

              {/* Trust Stats */}
              <div className="flex flex-wrap gap-6 justify-center lg:justify-start pt-4">
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center text-lg">🌾</div>
                  <div>
                    <p className="text-lg font-bold text-gray-900">1000+</p>
                    <p className="text-xs text-gray-500">Farmers Helped</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-full bg-yellow-100 flex items-center justify-center text-lg">🤖</div>
                  <div>
                    <p className="text-lg font-bold text-gray-900">24/7</p>
                    <p className="text-xs text-gray-500">AI Support</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-full bg-emerald-100 flex items-center justify-center text-lg">📈</div>
                  <div>
                    <p className="text-lg font-bold text-gray-900">95%</p>
                    <p className="text-xs text-gray-500">Accuracy</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right — Image */}
            <div className="relative flex-shrink-0">
              {/* Glow ring behind the image */}
              <div className="absolute inset-0 m-auto w-72 h-72 md:w-96 md:h-96 bg-gradient-to-br from-green-300/40 to-yellow-300/40 rounded-full blur-2xl" />
              <div className="relative">
                <div className="w-72 h-72 md:w-96 md:h-96 rounded-full bg-gradient-to-br from-green-100 to-yellow-50 p-2 shadow-2xl shadow-green-100">
                  <div className="w-full h-full rounded-full overflow-hidden bg-white flex items-center justify-center">
                    <Image
                      src="/images/logo.png"
                      alt="Agri-Ved Farming"
                      width={400}
                      height={400}
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>
                {/* Floating badges around the image */}
                <div className="absolute -top-2 -right-2 px-3 py-1.5 bg-white rounded-full shadow-lg border border-green-100 text-sm font-semibold text-green-700 animate-bounce" style={{ animationDuration: "3s" }}>
                  🌱 Smart Farming
                </div>
                <div className="absolute -bottom-2 -left-2 px-3 py-1.5 bg-white rounded-full shadow-lg border border-yellow-100 text-sm font-semibold text-yellow-700 animate-bounce" style={{ animationDuration: "3.5s" }}>
                  🌤️ Weather AI
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Services Section */}
      <section ref={addToFaders} className="py-16 px-6 md:px-20 bg-yellow-50 fade-up" id="services">
        <h2 className="text-3xl font-bold text-center text-green-700 mb-12">
          {t("servicesTitle")}
        </h2>
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          <div className="p-6 bg-white rounded-2xl shadow hover:shadow-lg transition border-t-4 border-yellow-500">
            <div className="text-4xl">🌽</div>
            <h3 className="mt-4 text-xl font-semibold text-green-700">{t("servicesCrop")}</h3>
            <p className="mt-2 text-gray-600">{t("servicesCropDesc")}</p>
          </div>

          <div className="p-6 bg-white rounded-2xl shadow hover:shadow-lg transition border-t-4 border-green-500">
            <div className="text-4xl">🤖</div>
            <h3 className="mt-4 text-xl font-semibold text-green-700">{t("servicesChatbot")}</h3>
            <p className="mt-2 text-gray-600">{t("servicesChatbotDesc")}</p>
          </div>

          <div className="p-6 bg-white rounded-2xl shadow hover:shadow-lg transition border-t-4 border-yellow-500">
            <div className="text-4xl">🌤️</div>
            <h3 className="mt-4 text-xl font-semibold text-green-700">{t("servicesWeather")}</h3>
            <p className="mt-2 text-gray-600">{t("servicesWeatherDesc")}</p>
          </div>

          <div className="p-6 bg-white rounded-2xl shadow hover:shadow-lg transition border-t-4 border-green-500">
            <div className="text-4xl">🌱</div>
            <h3 className="mt-4 text-xl font-semibold text-green-700">{t("servicesFertilizer")}</h3>
            <p className="mt-2 text-gray-600">{t("servicesFertilizerDesc")}</p>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section ref={addToFaders} className="py-16 px-6 md:px-20 bg-green-50 fade-up" id="contact">
        <h2 className="text-3xl font-bold text-center text-yellow-600 mb-8">{t("contactTitle")}</h2>
        <div className="text-center space-y-3">
          <p>📧 <span>{t("emailLabel")}</span> <a href="mailto:agrived@gmail.com" className="text-green-700 hover:underline">agrived@gmail.com</a></p>
          <p>📞 <span>{t("phoneLabel")}</span> <a href="tel:+911234567890" className="text-green-700 hover:underline">+91 12345 67890</a></p>
          <p>🏢 <span>{t("addressLabel")}</span> <span>{t("address")}</span></p>
        </div>
      </section>
    </div>
  );
}
