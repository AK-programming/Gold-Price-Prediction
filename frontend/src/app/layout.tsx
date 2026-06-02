import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "GoldSight AI — Intelligent Gold Price Forecasting",
  description:
    "AI-powered gold price forecasting dashboard featuring LSTM, Transformer, and ensemble models with SHAP explainability and attention visualization.",
  keywords: ["gold", "price", "forecast", "AI", "machine learning", "LSTM", "transformer"],
  authors: [{ name: "GoldSight AI" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
