/**
 * components/RecipeResults.jsx
 *
 * Renders the list of recipe cards returned from the backend.
 * The `results` shape matches what Flask's /api/recommend returns
 * after merging search + recommendation + user preference filtering.
 */

export function RecipeResults({ results, query, topQueryIngredients }) {
  if (!results.length) return null;

  // Guard against null query on first render
  const dishName = query?.dish_name ?? "";
  const cuisine  = query?.cuisine ?? "";

  return (
    <div>
      {/* Query summary */}
      <div className="mb-4 p-3 rounded" style={{ backgroundColor: "rgba(0,0,0,0.3)" }}>
        <p className="text-white mb-1">
          You searched for{" "}
          <strong className="text-capitalize">{dishName}</strong>
          {cuisine && (
            <> (<span className="text-capitalize">{cuisine}</span>)</>
          )}
        </p>

        {topQueryIngredients.length > 0 && (
          <div>
            <p className="text-white-50 small mb-1">
              Top 5 most distinctive ingredients of your dish:
            </p>
            <div className="d-flex flex-wrap gap-1">
              {topQueryIngredients.map((ing, i) => (
                <span
                  key={i}
                  className="badge"
                  style={{ backgroundColor: "rgba(255,255,255,0.85)", color: "#333" }}
                >
                  {ing}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <h2 className="h5 mb-3 text-white">
        {results.length} similar recipe{results.length !== 1 ? "s" : ""} from other cuisines
      </h2>

      <div className="row g-3">
        {results.map((recipe, i) => (
          <div key={recipe.id || recipe.fixed_url || i} className="col-md-6">
            <RecipeCard recipe={recipe} />
          </div>
        ))}
      </div>

      <div className="mt-4 text-center">
        <a href="/" className="btn btn-outline-light">
          ← Look for something else!
        </a>
      </div>
    </div>
  );
}

function RecipeCard({ recipe }) {
  const similarityScore =
    recipe.similarity_score > 0
      ? `${Math.round(recipe.similarity_score * 10000) / 10000}`
      : null;

  const cuisine =
    recipe.imputed_label ||
    recipe.cuisine_types?.join(", ") ||
    "Unknown cuisine";

  const shared      = recipe.shared_ingredients ?? [];
  const distinctive = recipe.distinctive_ingredients ?? recipe.ingred_weights ?? [];

  return (
    <div
      className="card h-100"
      style={{ backgroundColor: "rgba(255,255,255,0.92)" }}
    >
      <div className="card-body d-flex flex-column">
        <h3 className="card-title h6 mb-1">
          <a
            href={recipe.fixed_url || recipe.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-decoration-none"
          >
            {recipe.hed || recipe.label || recipe.title}
          </a>
        </h3>

        <p className="card-text text-muted small mb-2">
          {recipe.source} &mdash;{" "}
          <span className="text-capitalize">{cuisine}</span>
        </p>

        {similarityScore && (
          <p className="card-text small mb-2">
            <span className="text-muted">Similarity: </span>
            {similarityScore}
          </p>
        )}

        {/* Shared ingredients — highlighted green, shows why it matched */}
        {shared.length > 0 && (
          <div className="mb-2">
            <p className="small text-muted mb-1">
              Shared with your dish:
            </p>
            <div className="d-flex flex-wrap gap-1">
              {shared.map((ing, i) => (
                <span
                  key={i}
                  className="badge"
                  style={{ backgroundColor: "#d4edda", color: "#155724", border: "1px solid #c3e6cb" }}
                >
                  {ing}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Distinctive ingredients — shows what's different */}
        {distinctive.length > 0 && (
          <div className="mb-2">
            <p className="small text-muted mb-1">
              What makes it different:
            </p>
            <div className="d-flex flex-wrap gap-1">
              {distinctive.map((ing, i) => (
                <span
                  key={i}
                  className="badge bg-light text-dark border"
                >
                  {ing}
                </span>
              ))}
            </div>
          </div>
        )}

        {recipe.restriction_warning && (
          <div className="alert alert-warning py-1 px-2 small mb-2">
            {recipe.restriction_warning}
          </div>
        )}

        <a
          href={recipe.fixed_url || recipe.url}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-outline-primary btn-sm mt-auto"
        >
          View recipe ↗
        </a>
      </div>
    </div>
  );
}
