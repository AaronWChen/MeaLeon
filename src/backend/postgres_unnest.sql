-- unnest cuisines to do basic aggregation on cuisine count
SELECT
    rs.mealeon_id
    , rs.origin
    , c.cuisine
FROM recipe_scrapes rs
CROSS JOIN UNNEST(rs.cuisines) as c(cuisine);

-- unn
COPY (
    WITH staging AS (
        SELECT
            rs.mealeon_id
            , rs.origin
            , c.cuisine
        FROM recipe_scrapes rs
        CROSS JOIN UNNEST(rs.cuisines) AS c(cuisine)
    )

    SELECT
        origin
        , cuisine
        , COUNT(DISTINCT mealeon_id) AS num_recipes
    FROM staging
    GROUP BY
        origin
        , cuisine
) 
TO '/home/awchen/Repos/Projects/MeaLeon/data/processed/origin-cuisine-recipe-count.csv' DELIMITER ',' CSV HEADER;

