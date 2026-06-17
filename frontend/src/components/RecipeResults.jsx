/**
 * components/RecipeResults.jsx
 *
 * Renders the list of recipe cards returned from the backend.
 * The `results` shape matches what Flask's /api/recommend returns
 * after merging search + recommendation + user preference filtering.
 */

export function RecipeResults({ results, dishName, cuisine }) {
  if (!results.length) return null;

  return (
    <div>
      <h2 className="h5 mb-1">
        {results.length} similar recipe{results.length !== 1 ? "s" : ""} found
      </h2>
      <p className="text-muted small mb-3">
        You searched for <strong>{dishName}</strong>
        {cuisine && <> ({cuisine})</>} — showing cross-cuisine alternatives
      </p>
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
  const similarityScore = recipe.similarity_score > 0
    ? `${Math.round(recipe.similarity_score * 10000) / 10000}`
    : "N/A (Vespa index pending)";

  const cuisine = recipe.imputed_label
    || (recipe.cuisine_types?.join(", "))
    || "Unknown cuisine";

  const topIngredients = recipe.ingred_weights?.length
    ? recipe.ingred_weights
    : recipe.ingredient_names?.slice(0, 5) ?? [];

  return (
    <div className="card h-100">
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
          {recipe.source} &mdash; {cuisine}
        </p>

        <p className="card-text small mb-2">
          <span className="text-muted">Similarity: </span>
          {similarityScore}
        </p>

        {topIngredients.length > 0 && (
          <div className="mb-2">
            <p className="small text-muted mb-1">Distinctive ingredients:</p>
            <div className="d-flex flex-wrap gap-1">
              {topIngredients.map((ing, i) => (
                <span key={i} className="badge bg-light text-dark border">
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
