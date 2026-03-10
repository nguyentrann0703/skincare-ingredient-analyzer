import { useState } from "react";

const MOCK_PRODUCT = {
  name: "Moisturizing Barrier Cream",
  brand: "CeraVe",
  scanned_at: "2024-06-01T10:32:00Z",
  total_detected: 6,
  ingredients: [
    {
      name: "Water",
      rating: "Good",
      safety_label: "Safe",
      group: "no_concern",
      categories: ["Solvent"],
      description: "Universal solvent and base for most cosmetic formulas. No known concerns.",
    },
    {
      name: "Glycerin",
      rating: "Best",
      safety_label: "Very Safe",
      group: "no_concern",
      categories: ["Humectant", "Texture Enhancer"],
      description: "Highly effective humectant that draws moisture to skin. Well-tolerated across all skin types.",
    },
    {
      name: "Niacinamide",
      rating: "Best",
      safety_label: "Very Safe",
      group: "no_concern",
      categories: ["Antioxidant", "Humectant"],
      description: "Vitamin B3 derivative. Strengthens skin barrier, reduces pore appearance, brightens tone.",
    },
    {
      name: "Butylene Glycol",
      rating: "Good",
      safety_label: "Safe",
      group: "no_concern",
      categories: ["Humectant", "Texture Enhancer"],
      description: "Solvent and humectant. Helps other ingredients absorb better into skin.",
    },
    {
      name: "Phenoxyethanol",
      rating: "Good",
      safety_label: "Safe",
      group: "worth_knowing",
      categories: ["Preservative"],
      description: "Common preservative. Considered safe at concentrations up to 1% (EU/FDA standard). May cause mild irritation in very sensitive individuals.",
      note: "Preservative — safe within regulated limits",
    },
    {
      name: "Fragrance",
      rating: "Bad",
      safety_label: "Caution Advised",
      group: "potential_concern",
      categories: ["Irritant", "Fragrance: Synthetic and Natural"],
      description: "Collective term for fragrant compounds. Leading cause of contact allergy in cosmetics. Not recommended for sensitive or reactive skin.",
      note: "Common allergen — may trigger irritation or sensitization",
    },
  ],
};

const GROUP_CONFIG = {
  no_concern: {
    label: "No Concerns",
    icon: "✓",
    color: "#16a34a",
    bg: "#f0fdf4",
    border: "#bbf7d0",
    dot: "#22c55e",
  },
  worth_knowing: {
    label: "Worth Knowing",
    icon: "⚠",
    color: "#b45309",
    bg: "#fffbeb",
    border: "#fde68a",
    dot: "#f59e0b",
  },
  potential_concern: {
    label: "Potential Concerns",
    icon: "✕",
    color: "#dc2626",
    bg: "#fef2f2",
    border: "#fecaca",
    dot: "#ef4444",
  },
};

function IngredientRow({ ingredient, isLast }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = GROUP_CONFIG[ingredient.group];

  return (
    <div
      style={{
        borderBottom: isLast ? "none" : "1px solid #f1f5f9",
        padding: "12px 0",
      }}
    >
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        {/* Icon */}
        <span
          style={{
            width: 26,
            height: 26,
            borderRadius: "50%",
            background: cfg.bg,
            border: `1.5px solid ${cfg.border}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 11,
            fontWeight: 700,
            color: cfg.color,
            flexShrink: 0,
          }}
        >
          {cfg.icon}
        </span>

        {/* Name */}
        <span style={{ flex: 1, fontWeight: 500, color: "#0f172a", fontSize: 14 }}>
          {ingredient.name}
        </span>

        {/* Categories */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {ingredient.categories.slice(0, 2).map((cat) => (
            <span
              key={cat}
              style={{
                fontSize: 10,
                fontFamily: "'DM Mono', monospace",
                background: "#f1f5f9",
                color: "#64748b",
                padding: "2px 7px",
                borderRadius: 4,
                letterSpacing: "0.02em",
              }}
            >
              {cat}
            </span>
          ))}
        </div>

        {/* Safety label */}
        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: cfg.color,
            background: cfg.bg,
            padding: "3px 9px",
            borderRadius: 20,
            border: `1px solid ${cfg.border}`,
            whiteSpace: "nowrap",
            minWidth: 80,
            textAlign: "center",
          }}
        >
          {ingredient.safety_label}
        </span>

        {/* Expand arrow */}
        <span style={{ color: "#94a3b8", fontSize: 12, marginLeft: 4 }}>
          {expanded ? "▲" : "▼"}
        </span>
      </div>

      {/* Expanded description */}
      {expanded && (
        <div
          style={{
            marginTop: 10,
            marginLeft: 38,
            padding: "12px 14px",
            background: cfg.bg,
            borderRadius: 8,
            border: `1px solid ${cfg.border}`,
          }}
        >
          <p style={{ margin: 0, fontSize: 13, color: "#334155", lineHeight: 1.6 }}>
            {ingredient.description}
          </p>
          {ingredient.note && (
            <p
              style={{
                margin: "8px 0 0",
                fontSize: 12,
                color: cfg.color,
                fontStyle: "italic",
                fontWeight: 500,
              }}
            >
              ⓘ {ingredient.note}
            </p>
          )}
          <p
            style={{
              margin: "8px 0 0",
              fontSize: 11,
              color: "#94a3b8",
              fontFamily: "'DM Mono', monospace",
            }}
          >
            Source: Paula's Choice Ingredient Dictionary · Rating: {ingredient.rating}
          </p>
        </div>
      )}
    </div>
  );
}

function GroupSection({ group, ingredients }) {
  const cfg = GROUP_CONFIG[group];
  if (!ingredients.length) return null;

  return (
    <div style={{ marginBottom: 8 }}>
      {/* Group header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 0 6px",
          borderBottom: `2px solid ${cfg.border}`,
          marginBottom: 4,
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: cfg.dot,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: 12,
            fontWeight: 700,
            color: cfg.color,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
          }}
        >
          {cfg.label}
        </span>
        <span
          style={{
            fontSize: 11,
            color: "#94a3b8",
            fontFamily: "'DM Mono', monospace",
          }}
        >
          {ingredients.length} ingredient{ingredients.length > 1 ? "s" : ""}
        </span>
      </div>

      {ingredients.map((ing, i) => (
        <IngredientRow
          key={ing.name}
          ingredient={ing}
          isLast={i === ingredients.length - 1}
        />
      ))}
    </div>
  );
}

export default function IngredientAnalysis() {
  const product = MOCK_PRODUCT;

  const grouped = {
    no_concern: product.ingredients.filter((i) => i.group === "no_concern"),
    worth_knowing: product.ingredients.filter((i) => i.group === "worth_knowing"),
    potential_concern: product.ingredients.filter((i) => i.group === "potential_concern"),
  };

  const warnings = product.ingredients.filter((i) => i.group === "potential_concern");

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; background: #f8fafc; }
      `}</style>

      <div
        style={{
          fontFamily: "'DM Sans', sans-serif",
          minHeight: "100vh",
          background: "#f8fafc",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "center",
          padding: "32px 16px",
        }}
      >
        <div style={{ width: "100%", maxWidth: 520 }}>

          {/* Header */}
          <div style={{ marginBottom: 20 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 4,
              }}
            >
              <span
                style={{
                  fontSize: 10,
                  fontFamily: "'DM Mono', monospace",
                  color: "#94a3b8",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}
              >
                Analysis Result
              </span>
              <span
                style={{
                  fontSize: 10,
                  background: "#dbeafe",
                  color: "#1d4ed8",
                  padding: "1px 7px",
                  borderRadius: 10,
                  fontWeight: 600,
                }}
              >
                RAG-powered
              </span>
            </div>
            <h1
              style={{
                margin: 0,
                fontSize: 22,
                fontWeight: 700,
                color: "#0f172a",
                letterSpacing: "-0.02em",
              }}
            >
              {product.name}
            </h1>
            <p style={{ margin: "4px 0 0", fontSize: 13, color: "#64748b" }}>
              {product.brand} ·{" "}
              <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 12 }}>
                {product.total_detected} ingredients detected
              </span>
            </p>
          </div>

          {/* Summary bar */}
          <div
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 12,
              padding: "14px 18px",
              marginBottom: 16,
              display: "flex",
              gap: 0,
            }}
          >
            {Object.entries(GROUP_CONFIG).map(([key, cfg], i, arr) => (
              <div
                key={key}
                style={{
                  flex: 1,
                  textAlign: "center",
                  borderRight: i < arr.length - 1 ? "1px solid #f1f5f9" : "none",
                  padding: "0 12px",
                }}
              >
                <div
                  style={{
                    fontSize: 22,
                    fontWeight: 700,
                    color: cfg.color,
                    lineHeight: 1,
                  }}
                >
                  {grouped[key].length}
                </div>
                <div
                  style={{
                    fontSize: 10,
                    color: "#94a3b8",
                    marginTop: 3,
                    fontWeight: 500,
                    letterSpacing: "0.02em",
                  }}
                >
                  {cfg.label}
                </div>
              </div>
            ))}
          </div>

          {/* Warnings box */}
          {warnings.length > 0 && (
            <div
              style={{
                background: "#fef2f2",
                border: "1px solid #fecaca",
                borderRadius: 10,
                padding: "12px 16px",
                marginBottom: 16,
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: "#dc2626",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  marginBottom: 6,
                }}
              >
                ⚑ Warnings
              </div>
              {warnings.map((w) => (
                <div
                  key={w.name}
                  style={{
                    fontSize: 13,
                    color: "#7f1d1d",
                    lineHeight: 1.5,
                    marginBottom: 2,
                  }}
                >
                  <strong>{w.name}</strong> — {w.note}
                </div>
              ))}
            </div>
          )}

          {/* Ingredient groups */}
          <div
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 12,
              padding: "16px 18px",
              marginBottom: 16,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: "#94a3b8",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                marginBottom: 14,
              }}
            >
              Detected Ingredients
            </div>

            <GroupSection group="no_concern" ingredients={grouped.no_concern} />
            <GroupSection group="worth_knowing" ingredients={grouped.worth_knowing} />
            <GroupSection group="potential_concern" ingredients={grouped.potential_concern} />
          </div>

          {/* Footer */}
          <div style={{ textAlign: "center" }}>
            <p
              style={{
                fontSize: 11,
                color: "#94a3b8",
                fontFamily: "'DM Mono', monospace",
                margin: 0,
              }}
            >
              Data source: Paula's Choice Ingredient Dictionary
              <br />
              This analysis is informational only — consult a dermatologist for medical advice.
            </p>
          </div>

        </div>
      </div>
    </>
  );
}
