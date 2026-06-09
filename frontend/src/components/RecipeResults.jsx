/**
 * components/RecipeResults.jsx
 *
 * Renders the list of recipe cards returned from the backend.
 * The `results` shape matches what Flask's /api/recommend returns
 * after merging search + recommendation + user preference filtering.
 */

export function RecipeResults({ results }) {
  if (!results.length) return null;

  return (
    <div>
      <h2 className="h5 mb-3">
        {results.length} recipe{results.length !== 1 ? "s" : ""} found
      </h2>
      <div className="row g-3">
        {results.map((recipe) => (
          <div key={recipe.id} className="col-md-6 col-lg-4">
            <RecipeCard recipe={recipe} />
          </div>
        ))}
      </div>
    </div>
  );
}

function RecipeCard({ recipe }) {
  return (
    <div className="card h-100">
      {recipe.image_url && (
        <img
          src={recipe.image_url}
          alt={recipe.label}
          className="card-img-top"
          style={{ objectFit: "cover", height: 160 }}
        />
      )}
      <div className="card-body d-flex flex-column">
        <h3 className="card-title h6">{recipe.label}</h3>
        <p className="card-text text-muted small mb-1">{recipe.source}</p>

        {/* Similarity score — comes from the recommendation service */}
        {recipe.similarity_score != null && (
          <p className="card-text small mb-2">
            Match: {Math.round(recipe.similarity_score * 100)}%
          </p>
        )}

        {/* Diet/health labels — comes from Edamam via search service */}
        {recipe.diet_labels?.length > 0 && (
          <div className="d-flex flex-wrap gap-1 mb-2">
            {recipe.diet_labels.slice(0, 3).map((label) => (
              <span key={label} className="badge bg-secondary">
                {label}
              </span>
            ))}
          </div>
        )}

        {/* Restriction warning — added by Flask if the recipe contains
            something the user is allergic to but was still returned
            (e.g. soft filter mode) */}
        {recipe.restriction_warning && (
          <div className="alert alert-warning py-1 px-2 small mb-2">
            {recipe.restriction_warning}
          </div>
        )}

        <a
          href={recipe.url}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-outline-primary btn-sm mt-auto"
        >
          View recipe
        </a>
      </div>
    </div>
  );
}
