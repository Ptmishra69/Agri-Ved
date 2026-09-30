"use client";

import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import { ProtectedRoute } from "@/components/AuthContext";

const API_KEY = "a701e399628eb468b780be2d8f07cb8b";

function WeatherContent() {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  useEffect(() => {
    const fetchWeather = async () => {
      const LAT = localStorage.getItem("latitude");
      const LON = localStorage.getItem("longitude");

      if (!LAT || !LON) {
        alert("⚠️ No location data found! Please sign up or login again.");
        return;
      }

      try {
        const response = await fetch(
          `https://api.openweathermap.org/data/2.5/onecall?lat=${LAT}&lon=${LON}&exclude=current,minutely,hourly,alerts&units=metric&appid=${API_KEY}`
        );
        const data = await response.json();

        const labels = data.daily.slice(0, 7).map((d: any) =>
          new Date(d.dt * 1000).toLocaleDateString("en-US", { weekday: "short" })
        );

        const tempData = data.daily.slice(0, 7).map((d: any) => d.temp.day);
        const rainData = data.daily.slice(0, 7).map((d: any) => d.rain || 0);
        const humidityData = data.daily.slice(0, 7).map((d: any) => d.humidity);

        renderChart(labels, tempData, rainData, humidityData);
      } catch (err) {
        console.error("Weather fetch failed:", err);
        alert("⚠️ Unable to fetch weather data. Check API key or location.");
      }
    };

    fetchWeather();

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
    };
  }, []);

  const renderChart = (labels: string[], tempData: number[], rainData: number[], humidityData: number[]) => {
    if (!chartRef.current) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const ctx = chartRef.current.getContext("2d");
    if (!ctx) return;

    chartInstanceRef.current = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Rainfall (mm)",
            data: rainData,
            type: "bar",
            backgroundColor: "rgba(54, 162, 235, 0.6)",
            borderColor: "rgba(54, 162, 235, 1)",
            borderWidth: 1,
            yAxisID: "y-rain"
          },
          {
            label: "Temperature (°C)",
            data: tempData,
            type: "line",
            borderColor: "rgba(255, 99, 132, 1)",
            backgroundColor: "rgba(255, 99, 132, 0.2)",
            tension: 0.4,
            fill: false,
            yAxisID: "y-temp"
          },
          {
            label: "Humidity (%)",
            data: humidityData,
            type: "line",
            borderColor: "rgba(75, 192, 192, 1)",
            backgroundColor: "rgba(75, 192, 192, 0.2)",
            tension: 0.4,
            fill: false,
            yAxisID: "y-humidity"
          }
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        stacked: false,
        plugins: {
          legend: { position: "top" },
          tooltip: { mode: "index", intersect: false },
        },
        scales: {
          "y-temp": {
            type: "linear",
            position: "left",
            title: { display: true, text: "Temperature (°C)" },
          },
          "y-rain": {
            type: "linear",
            position: "right",
            title: { display: true, text: "Rainfall (mm)" },
            grid: { drawOnChartArea: false }
          },
          "y-humidity": {
            type: "linear",
            position: "right",
            title: { display: true, text: "Humidity (%)" },
            grid: { drawOnChartArea: false }
          }
        }
      }
    });
  };

  return (
    <div className="pt-10 px-6 min-h-[80vh]">
      <h2 className="text-2xl font-bold text-center mb-6 text-gray-800">🌦️ 7-Day Weather Forecast</h2>
      <div className="max-w-4xl mx-auto bg-white p-6 shadow-lg rounded-xl">
        <canvas ref={chartRef} className="w-full"></canvas>
      </div>
    </div>
  );
}

export default function Weather() {
  return (
    <ProtectedRoute>
      <WeatherContent />
    </ProtectedRoute>
  );
}
