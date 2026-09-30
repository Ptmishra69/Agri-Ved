"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useLanguage } from "./LanguageContext";
import { useAuth } from "./AuthContext";

export default function Navbar() {
  const { lang, setLang, t } = useLanguage();
  const { user, logout } = useAuth();
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  const navLinks = [
    { href: "/", label: t("home") },
    { href: "/chatbot", label: t("chatbot") },
    { href: "/weather", label: t("weather") },
    { href: "/#services", label: t("services") },
    { href: "/#contact", label: t("contact") },
  ];

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href.split("#")[0]) && href.split("#")[0] !== "/";
  };

  return (
    <header
      className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/80 backdrop-blur-xl shadow-lg border-b border-green-100"
          : "bg-gradient-to-r from-green-50/90 to-yellow-50/90 backdrop-blur-md"
      }`}
    >
      <nav className="flex items-center justify-between px-6 lg:px-10 py-3">
        {/* Logo + Brand */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="relative">
            <div className="absolute -inset-1 bg-gradient-to-r from-green-400 to-yellow-400 rounded-full opacity-0 group-hover:opacity-60 blur transition-opacity duration-300" />
            <Image
              src="/images/logo.png"
              alt="Agri-Ved logo"
              width={44}
              height={44}
              className="relative w-11 h-11 rounded-full ring-2 ring-green-200 group-hover:ring-green-400 transition-all"
            />
          </div>
          <span className="text-xl font-bold bg-gradient-to-r from-green-700 to-green-500 bg-clip-text text-transparent hidden sm:block">
            Agri-Ved
          </span>
        </Link>

        {/* Desktop Nav Links */}
        <ul className="hidden lg:flex items-center gap-0.5">
          {navLinks.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive(link.href)
                    ? "text-green-700 bg-green-100/70"
                    : "text-gray-600 hover:text-green-700 hover:bg-green-50"
                }`}
              >
                {link.label}
                {isActive(link.href) && (
                  <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-5 h-0.5 bg-green-500 rounded-full" />
                )}
              </Link>
            </li>
          ))}
        </ul>

        {/* Right Side */}
        <div className="flex items-center gap-3">
          {/* Language Switcher */}
          <div className="relative">
            <select
              value={lang}
              onChange={(e) => setLang(e.target.value as any)}
              className="appearance-none pl-3 pr-7 py-1.5 rounded-lg bg-white/70 border border-green-200 text-green-800 text-xs font-semibold tracking-wide uppercase cursor-pointer hover:border-green-400 focus:outline-none focus:ring-2 focus:ring-green-300 transition-all"
            >
              <option value="en">EN</option>
              <option value="hi">हि</option>
              <option value="pa">ਪੰ</option>
            </select>
            <span className="absolute right-2 top-1/2 -translate-y-1/2 text-green-500 text-[10px] pointer-events-none">▼</span>
          </div>

          {/* Auth Buttons — Desktop */}
          <div className="hidden lg:flex items-center gap-2">
            {user ? (
              <>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-green-50 rounded-full border border-green-200">
                  <div className="w-7 h-7 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-white text-xs font-bold shadow-sm">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                  <span className="text-sm font-medium text-green-800 max-w-[100px] truncate">
                    {user.name}
                  </span>
                </div>
                <button
                  onClick={logout}
                  className="px-3 py-1.5 text-xs font-medium text-red-500 border border-red-200 rounded-lg hover:bg-red-50 hover:border-red-300 transition-all"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link
                  href="/login"
                  className="px-4 py-2 text-sm font-medium text-green-700 border border-green-300 rounded-lg hover:bg-green-50 hover:border-green-400 transition-all"
                >
                  {t("login")}
                </Link>
                <Link
                  href="/login"
                  className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-green-500 to-emerald-600 rounded-lg hover:from-green-600 hover:to-emerald-700 shadow-md shadow-green-200 hover:shadow-lg hover:shadow-green-300 transition-all"
                >
                  {t("signup")}
                </Link>
              </>
            )}
          </div>

          {/* Mobile Hamburger */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="lg:hidden flex flex-col gap-1.5 p-2 rounded-lg hover:bg-green-50 transition-colors"
            aria-label="Toggle menu"
          >
            <span className={`w-5 h-0.5 bg-green-700 rounded-full transition-all duration-300 ${mobileOpen ? "rotate-45 translate-y-2" : ""}`} />
            <span className={`w-5 h-0.5 bg-green-700 rounded-full transition-all duration-300 ${mobileOpen ? "opacity-0" : ""}`} />
            <span className={`w-5 h-0.5 bg-green-700 rounded-full transition-all duration-300 ${mobileOpen ? "-rotate-45 -translate-y-2" : ""}`} />
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      <div
        className={`lg:hidden overflow-hidden transition-all duration-300 ${
          mobileOpen ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <div className="px-6 pb-5 pt-2 bg-white/95 backdrop-blur-xl border-t border-green-100 space-y-1">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`block px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive(link.href)
                  ? "text-green-700 bg-green-100/70"
                  : "text-gray-600 hover:text-green-700 hover:bg-green-50"
              }`}
            >
              {link.label}
            </Link>
          ))}

          <div className="pt-3 border-t border-green-100">
            {user ? (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-white text-sm font-bold">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                  <span className="text-sm font-medium text-green-800">{user.name}</span>
                </div>
                <button
                  onClick={logout}
                  className="px-3 py-1.5 text-xs font-medium text-red-500 border border-red-200 rounded-lg hover:bg-red-50"
                >
                  Logout
                </button>
              </div>
            ) : (
              <div className="flex gap-2">
                <Link href="/login" className="flex-1 text-center py-2.5 text-sm font-medium text-green-700 border border-green-300 rounded-lg hover:bg-green-50">
                  {t("login")}
                </Link>
                <Link href="/login" className="flex-1 text-center py-2.5 text-sm font-medium text-white bg-gradient-to-r from-green-500 to-emerald-600 rounded-lg shadow-md">
                  {t("signup")}
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
