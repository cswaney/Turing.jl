using Distributions, Gadfly, Random
using Turing, MCMCChains
Turing.turnprogress(false)  # turn off the progress monitor


Random.seed!(3)


# Construct the data points
N = 30
μs = [[0., 0.], [5., 5.]]
x1 = rand(MvNormal(μs[1], 1.), N)
x2 = rand(MvNormal(μs[2], 1.), N)
x = hcat([x1, x2]...)
plot(x=x[1,:], y=x[2,:])


# Define the probabilistic model
@model GaussianMixtureModel(x) = begin
    K, N = size(x)

    # Global variables
    μ = 0.
    σ = 1.
    μ1 ~ Normal(μ, σ)
    μ2 ~ Normal(μ, σ)
    locs = [μ1, μ2]
    α = 1.
    weights ~ Dirichlet(K, α)  # latent
    # weights = ones(K) / K;  # observed

    # Local context
    z = Vector{Int64}(undef, N)
    for i in 1:N
        z[i] ~ Categorical(weights)
        x[:,i] ~ MvNormal([locs[z[i]], locs[z[i]]], 1.)
    end
    return z  # only return *latent* local variables
end

gmm_model = GaussianMixtureModel(x);


# Perform MCMC inference
gmm_sampler = Gibbs(
    PG(100, :z),  # Particle Gibbs (discrete)
    HMC(0.05, 10, :μ1, :μ2, :weights)  # Hamiltonian Monte Carlo (continuous)
    # HMC(0.05, 10, :μ1, :μ2, )  # Hamiltonian Monte Carlo (continuous)
);
@time chain = sample(gmm_model, gmm_sampler, 100);







ids = findall(map(name -> occursin("μ", name), names(chain)));
p=plot(chain[:, ids, :], legend=true, labels = ["Mu 1" "Mu 2"], colordim=:parameter)


chain = chain[:, :, 1];

function predict(x, y, w, μ)
    # Use log-sum-exp trick for numeric stability.
    return Turing.logaddexp(
        log(w[1]) + logpdf(MvNormal([μ[1], μ[1]], 1.), [x, y]),
        log(w[2]) + logpdf(MvNormal([μ[2], μ[2]], 1.), [x, y])
    )
end

contour(range(-5, stop = 3), range(-6, stop = 2),
    (x, y) -> predict(x, y, [0.5, 0.5], [mean(chain[:μ1].value), mean(chain[:μ2].value)])
)
scatter!(x[1,:], x[2,:], legend = false, title = "Synthetic Dataset")


assignments = collect(skipmissing(mean(chain[:k].value, dims=1).data))
scatter(x[1,:], x[2,:],
    legend = false,
    title = "Assignments on Synthetic Dataset",
    zcolor = assignments)
