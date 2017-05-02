import Base.string, Base.isequal, Base.==, Base.hash
import Base.getindex, Base.setindex!, Base.push!
import Base.rand, Base.show, Base.isnan

###########
# VarName #
###########
immutable VarName
  csym      ::    Symbol        # symbol generated in compilation time
  sym       ::    Symbol        # variable symbol
  indexing  ::    String        # indexing
  counter   ::    Int           # counter of same {csym, uid}
end

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)
hash(vn::VarName) = hash(uid(vn))

isequal(x::VarName, y::VarName) = uid(x) == uid(y)
==(x::VarName, y::VarName)      = isequal(x, y)

Base.string(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
Base.string(vns::Vector{VarName}) = replace(string(map(vn -> string(vn), vns)), "String", "")

sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing)    # the uid which is only available at compile time

###########
# VarInfo #
###########

type VarInfo
  idcs        ::    Dict{VarName, Int}
  vns         ::    Vector{VarName}
  ranges      ::    Vector{UnitRange{Int}}
  vals        ::    Vector{Vector{Real}}
  dists       ::    Vector{Distribution}
  gids        ::    Vector{Int}   # group ids
  trans       ::    Vector{Bool}
  logp        ::    Real
  logw        ::    Real          # NOTE: importance weight when sampling from the prior.
  index       ::    Int           # index of current randomness
  num_produce ::    Int           # num of produce calls from trace, each produce corresponds to an observe.
  VarInfo() = new(
    Dict{VarName, Int}(),
    Vector{VarName}(),
    Vector{UnitRange{Int}}(),
    Vector{Vector{Real}}(),
    Vector{Distribution}(),
    Vector{Int}(),
    Vector{Bool}(),
    0.0,0.0,
    0,
    0
  )
end

typealias VarView Union{Int,UnitRange,Vector{Int}}

getidx(vi::VarInfo, vn::VarName) = vi.idcs[vn]

getrange(vi::VarInfo, vn::VarName) = vi.ranges[getidx(vi, vn)]

getval(vi::VarInfo, vn::VarName)       = vi.vals[end][getrange(vi, vn)]
setval!(vi::VarInfo, val, vn::VarName) = vi.vals[end][getrange(vi, vn)] = val

getval(vi::VarInfo, view::VarView)       = vi.vals[end][view]
setval!(vi::VarInfo, val, view::VarView) = vi.vals[end][view] = val

getsym(vi::VarInfo, vn::VarName) = vi.vns[getidx(vi, vn)].sym

getdist(vi::VarInfo, vn::VarName) = vi.dists[getidx(vi, vn)]

getgid(vi::VarInfo, vn::VarName) = vi.gids[getidx(vi, vn)]

setgid!(vi::VarInfo, gid, vn::VarName) = vi.gids[getidx(vi, vn)] = gid

istransformed(vi::VarInfo, vn::VarName) = vi.trans[getidx(vi, vn)]

# X -> R for all variables associated with given sampler
function link(_vi, spl)
  vi = deepcopy(_vi)
  gvns = groupvns(vi, spl)
  for vn in gvns
    dist = getdist(vi, vn)
    setval!(vi, vectorize(dist, link(dist, reconstruct(dist, getval(vi, vn)))), vn)
    vi.trans[getidx(vi, vn)] = true
  end
  vi
end

# R -> X for all variables associated with given sampler
function invlink(_vi, spl)
  vi = deepcopy(_vi)
  gvns = groupvns(vi, spl)
  for vn in gvns
    dist = getdist(vi, vn)
    setval!(vi, vectorize(dist, invlink(dist, reconstruct(dist, getval(vi, vn)))), vn)
    vi.trans[getidx(vi, vn)] = false
  end
  vi
end

function cleandual!(vi::VarInfo)
  for vn in keys(vi)
    range = getrange(vi, vn)
    vi[range] = realpart(vi[range])
  end
  vi.logp = realpart(vi.logp)
  vi.logw = realpart(vi.logw)
end

vns(vi::VarInfo) = Set(keys(vi.idcs))            # get all vns
syms(vi::VarInfo) = map(vn -> vn.sym, vns(vi))  # get all symbols

# The default getindex & setindex!() for get & set values
# NOTE: vi[vn] will always transform the variable to its original space and Julia type
Base.getindex(vi::VarInfo, vn::VarName) = begin
  @assert haskey(vi, vn) "[Turing] attempted to replay unexisting variables in VarInfo"
  dist = getdist(vi, vn)
  r = reconstruct(dist, getval(vi, vn))
  r = istransformed(vi, vn) ? invlink(dist, r) : r
end

# Base.setindex!(vi::VarInfo, r, vn::VarName) = begin
#   dist = getdist(vi, vn)
#   setval!(vi, vectorize(dist, r), vn)
# end

# NOTE: vi[view] will just return what insdie vi (no transformations applied)
Base.getindex(vi::VarInfo, view::VarView)       = getval(vi, view)
Base.setindex!(vi::VarInfo, val, view::VarView) = setval!(vi, val, view)

Base.keys(vi::VarInfo) = keys(vi.idcs)

Base.haskey(vi::VarInfo, vn::VarName) = haskey(vi.idcs, vn)

Base.show(io::IO, vi::VarInfo) = begin
  vi_str = """
  /=======================================================================
  | VarInfo
  |-----------------------------------------------------------------------
  | Varnames  :   $(string(vi.vns))
  | Range     :   $(vi.ranges)
  | Vals      :   $(vi.vals)
  | GIDs      :   $(vi.gids)
  | Trans?    :   $(vi.trans)
  | Logp      :   $(vi.logp)
  | Logw      :   $(vi.logw)
  | Index     :   $(vi.index)
  | #produce  :   $(vi.num_produce)
  \\=======================================================================
  """
  print(io, vi_str)
end

#################################
# Utility functions for VarInfo #
#################################

expand!(vi::VarInfo) = push!(vi.vals, deepcopy(vi.vals[end]))
last(_vi::VarInfo) = begin
  vi = deepcopy(_vi)
  splice!(vi.vals, 1:length(vi.vals)-1)
  vi
end

# Get all indices of variables belonging to gid or 0
groupidcs(vi::VarInfo) = groupidcs(vi, nothing)
groupidcs(vi::VarInfo, spl::Void) = filter(i -> vi.gids[i] == 0 || vi.gids[i] == 0, 1:length(vi.gids))
groupidcs(vi::VarInfo, spl::Sampler) =
  filter(i ->
    (vi.gids[i] == spl.alg.gid || vi.gids[i] == 0) && (isempty(spl.alg.space) || vi.vns[i].sym in spl.alg.space),
    1:length(vi.gids)
  )

# Get all values of variables belonging to gid or 0
groupvals(vi::VarInfo) = groupvals(vi, nothing)
groupvals(vi::VarInfo, spl::Union{Void, Sampler}) = map(i -> vi[vi.ranges[i]], groupidcs(vi, spl))

# Get all vns of variables belonging to gid or 0
groupvns(vi::VarInfo) = groupvns(vi, nothing)
groupvns(vi::VarInfo, spl::Union{Void, Sampler}) = map(i -> vi.vns[i], groupidcs(vi, spl))

# Get all vns of variables belonging to gid or 0
getranges(vi::VarInfo, spl::Sampler) = union(map(i -> vi.ranges[i], groupidcs(vi, spl))...)

retain!(vi::VarInfo, n_retain::Int) = retain!(vi, n_retain, nothing)
retain!(vi::VarInfo, n_retain::Int, spl::Union{Void, Sampler}) = begin
  gidcs = groupidcs(vi, spl)

  # Set all corresponding entries to NaN
  l = length(gidcs)
  for i = l:-1:(n_retain + 1),  # for each variable (in reversed order)
      j = vi.ranges[gidcs[i]]   # for each index of variable range
    vi[j] = NaN
  end

  vi
end

immutable Var
  vn    ::  VarName
  val   ::  Vector{Real}
  dist  ::  Distribution
  gid   ::  Int
end

# Add a new entry to VarInfo
push!(vi::VarInfo, v::Var) = begin
  vn, val, dist, gid = v.vn, v.val, v.dist, v.gid

  @assert ~(vn in vns(vi)) "[push!] attempt to add an exisitng variable $(sym(vn)) ($(vn)) to VarInfo (keys=$(keys(vi))) with dist=$dist, gid=$gid"

  if isempty(vi.vals) push!(vi.vals, Vector{Real}()) end

  vi.idcs[vn] = length(vi.idcs) + 1
  push!(vi.vns, vn)
  l = length(vi.vals[end]); n = length(val)
  push!(vi.ranges, l+1:l+n)
  append!(vi.vals[end], val)
  push!(vi.dists, dist)
  push!(vi.gids, gid)
  push!(vi.trans, false)

  vi
end

# This method is use to generate a new VarName with the right count
VarName(vi::VarInfo, csym::Symbol, sym::Symbol, indexing::String) = begin
  # TODO: update this method when implementing the sanity check
  VarName(csym, sym, indexing, 1)
end

#######################################
# Rand & replaying method for VarInfo #
#######################################

# Check if a vn is set to NaN (by retain!)
isnan(vi::VarInfo, vn::VarName) = any(isnan(getval(vi, vn)))

# Sanity check for VarInfo.index
checkindex(vn::VarName, vi::VarInfo) = checkindex(vn, vi, nothing)
checkindex(vn::VarName, vi::VarInfo, spl::Union{Void, Sampler}) = begin
  vn_index = groupvns(vi, spl)[vi.index]
  @assert vn_index == vn "[Turing]: sanity check for VarInfo.index failed: vn_index=$vn_index, vi.index=$(vi.index), vn_now=$(vn)"
end

# Initialize VarInfo, i.e. sampling from priors
nwevar!(vi::VarInfo, vn::VarName, dist::Distribution) = nwevar!(vi, vn, dist, 0)
nwevar!(vi::VarInfo, vn::VarName, dist::Distribution, gid::Int) = begin
  @assert ~haskey(vi, vn) "[Turing] attempted to initialize existing variables in VarInfo"
  r = rand(dist)
  push!(vi, Var(vn, vectorize(dist, r), dist, gid))
  r
end

updategid!(vi, vn, spl) = begin
  if ~isempty(spl.alg.space) && getgid(vi, vn) == 0 && getsym(vi, vn) in spl.alg.space
    setgid!(vi, spl.alg.gid, vn)
  end
end

# This method is called when sampler is missing
# NOTE: this used for initialize VarInfo, i.e. vi = model()
# NOTE: this method is also used by IS
randr(vi::VarInfo, vn::VarName, dist::Distribution) = randr(vi, vn, dist, false)
randr(vi::VarInfo, vn::VarName, dist::Distribution, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    nwevar!(vi, vn, dist)
  else
    if count checkindex(vn, vi) end
    r = vi[vn]
    vi.logp += logpdf(dist, r, istransformed(vi, vn))
    r
  end
end

# Random with replaying
# NOTE: this method is used by PG
randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler) = randr(vi, vn, dist, spl, false)
randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Sampler, count::Bool) = begin
  vi.index = count ? vi.index + 1 : vi.index
  if ~haskey(vi, vn)
    nwevar!(vi, vn, dist, spl.alg.gid)
  elseif isnan(vi, vn)
    r = rand(dist)
    setval!(vi, vectorize(dist, r), vn)
    r
  else
    if count checkindex(vn, vi, spl) end
    updategid!(vi, vn, spl)
    vi[vn]
  end
end
