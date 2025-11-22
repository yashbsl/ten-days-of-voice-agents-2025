"use client";

export default function ThemeStyles() {
  // Write whatever CSS you want here.
  // This will ONLY render on the client, so no hydration mismatch.
  const css = `
    :root {
      --primary: #002cf2;
      --primary-hover: color-mix(in srgb, #002cf2 80%, #000);
    }
  `;

  return <style>{css}</style>;
}
