"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthContext";

const API_BASE = "http://localhost:5005";

interface LocationOption {
  name: string;
}

export default function Login() {
  const router = useRouter();
  const { login, signup } = useAuth();
  const [isSignIn, setIsSignIn] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Sign In State
  const [signInPhone, setSignInPhone] = useState("");
  const [signInPassword, setSignInPassword] = useState("");

  // Sign Up State
  const [signUpName, setSignUpName] = useState("");
  const [signUpPhone, setSignUpPhone] = useState("");
  const [signUpPassword, setSignUpPassword] = useState("");

  // Location (SHC schema: State → District → Block → Village)
  const [states, setStates] = useState<LocationOption[]>([]);
  const [selectedState, setSelectedState] = useState("");
  const [district, setDistrict] = useState("");
  const [block, setBlock] = useState("");
  const [village, setVillage] = useState("");

  // SHC
  const [hasShc, setHasShc] = useState("");
  const [shcId, setShcId] = useState("");

  // Load states on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/location/states`)
      .then(res => res.json())
      .then(data => setStates(data))
      .catch(() => setStates([]));
  }, []);

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(signInPhone, signInPassword);
      router.push("/");
    } catch (err: any) {
      setError(err.message || "Login failed. Check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      // Geocode for lat/lon
      let lat: number | null = null;
      let lon: number | null = null;
      const locationQuery = [village, block, district, selectedState, "India"].filter(Boolean).join(", ");

      try {
        const geoRes = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(locationQuery)}`
        );
        const geoData = await geoRes.json();
        if (geoData.length > 0) {
          lat = parseFloat(geoData[0].lat);
          lon = parseFloat(geoData[0].lon);
        }
      } catch {
        // Geocoding failed silently — coords will be null
      }

      await signup({
        name: signUpName,
        phone: signUpPhone,
        password: signUpPassword,
        state: selectedState,
        district: district,
        block: block || null,
        village: village || null,
        latitude: lat,
        longitude: lon,
        shc_id: hasShc === "yes" ? shcId : null,
      });

      setError("");
      alert("✅ Account created! Please sign in.");
      setIsSignIn(true);
    } catch (err: any) {
      setError(err.message || "Signup failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleShcChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    setHasShc(val);
    if (val === "no") {
      window.open("https://soilhealth.dac.gov.in/home", "_blank");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-[90vh] bg-gradient-to-br from-yellow-50 to-green-50">
      <div className="bg-white w-full max-w-md p-8 rounded-2xl shadow-xl fade-in">
        <h2 className="text-2xl font-bold text-green-700 text-center mb-6">
          {isSignIn ? "Sign In" : "Sign Up"}
        </h2>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
            ⚠️ {error}
          </div>
        )}

        {isSignIn ? (
          <form onSubmit={handleSignIn} className="flex flex-col gap-4">
            <label className="text-sm text-gray-700">Phone Number</label>
            <input
              type="tel"
              placeholder="Enter your phone number"
              required
              value={signInPhone}
              onChange={e => setSignInPhone(e.target.value)}
              className="px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
            />

            <label className="text-sm text-gray-700">Password</label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                required
                value={signInPassword}
                onChange={e => setSignInPassword(e.target.value)}
                className="w-full px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-2 text-green-600 text-sm"
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-green-500 text-white font-medium rounded-lg hover:bg-green-600 transition mt-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Signing in..." : "Sign In"}
            </button>
          </form>
        ) : (
          <form onSubmit={handleSignUp} className="flex flex-col gap-4">
            <label className="text-sm text-gray-700">Full Name</label>
            <input
              type="text"
              placeholder="Enter your full name"
              required
              value={signUpName}
              onChange={e => setSignUpName(e.target.value)}
              className="px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
            />

            <label className="text-sm text-gray-700">Phone Number</label>
            <input
              type="tel"
              placeholder="Enter your phone number"
              required
              value={signUpPhone}
              onChange={e => setSignUpPhone(e.target.value)}
              className="px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
            />

            {/* Location: State → District → Block → Village */}
            <div className="p-4 bg-green-50/50 rounded-xl border border-green-100 space-y-3">
              <p className="text-sm font-semibold text-green-700">📍 Location (as per Soil Health Card)</p>

              <label className="text-sm text-gray-700">State / UT *</label>
              <select
                required
                value={selectedState}
                onChange={e => { setSelectedState(e.target.value); setDistrict(""); setBlock(""); setVillage(""); }}
                className="w-full px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none bg-white"
              >
                <option value="" disabled>-- Select State --</option>
                {states.map(s => (
                  <option key={s.name} value={s.name}>{s.name}</option>
                ))}
              </select>

              <label className="text-sm text-gray-700">District *</label>
              <input
                type="text"
                placeholder="Enter your district"
                required
                value={district}
                onChange={e => setDistrict(e.target.value)}
                className="w-full px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
              />

              <label className="text-sm text-gray-700">Block / Tehsil</label>
              <input
                type="text"
                placeholder="Enter your block or tehsil"
                value={block}
                onChange={e => setBlock(e.target.value)}
                className="w-full px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
              />

              <label className="text-sm text-gray-700">Village</label>
              <input
                type="text"
                placeholder="Enter your village name"
                value={village}
                onChange={e => setVillage(e.target.value)}
                className="w-full px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
              />
            </div>

            {/* SHC */}
            <label className="text-sm text-gray-700">Do you have a Soil Health Card (SHC)?</label>
            <select
              required
              value={hasShc}
              onChange={handleShcChange}
              className="px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
            >
              <option value="" disabled>Select an option</option>
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>

            {hasShc === "yes" && (
              <>
                <label className="text-sm text-gray-700">SHC ID</label>
                <input
                  type="text"
                  placeholder="e.g. SHC123"
                  value={shcId}
                  onChange={e => setShcId(e.target.value)}
                  className="px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
                />
              </>
            )}

            <label className="text-sm text-gray-700">Password</label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Create a password"
                required
                value={signUpPassword}
                onChange={e => setSignUpPassword(e.target.value)}
                className="w-full px-3 py-2 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-400 focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-2 text-green-600 text-sm"
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-yellow-500 text-white font-medium rounded-lg hover:bg-yellow-600 transition mt-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Creating account..." : "Sign Up"}
            </button>
          </form>
        )}

        <div className="text-center mt-6 text-sm">
          <span>{isSignIn ? "Don't have an account? " : "Already have an account? "}</span>
          <button
            onClick={() => { setIsSignIn(!isSignIn); setError(""); }}
            className="text-green-600 font-semibold hover:underline"
          >
            {isSignIn ? "Sign Up" : "Sign In"}
          </button>
        </div>
      </div>
    </div>
  );
}
