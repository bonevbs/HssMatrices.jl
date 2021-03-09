### Visualization routines to aid development
# This entire implementation is extremely hacky...
# Ideally this should be eventually replaced by something based on RecipesBase
#
# Written by Boris Bonev, Nov. 2020

## routine for plotting ranks
function plotranks(hssA::HssMatrix; cutoff_level=3)
  m,n = size(hssA)
  aspect = m/n
  plot(yflip=true, showaxis=false, size = (400, 400*aspect))
  xticks = [1]; yticks = [1]
  yticks, xticks = _plotranks!(hssA,1,1,xticks,yticks,0,cutoff_level)
  append!(yticks, n+1)
  append!(xticks, m+1)
  plot!(aspect_ratio=:equal)
  plot_ref = plot!(xticks=xticks, yticks=yticks, xmirror=true)
  return plot_ref
end

# auxiliary routine for drawing a rectangle
_rectangle(w, h, x, y) = Shape([0,w,w,0] .+ x , [0,0,h,h] .+ y)

# recursive plotting routine
function _plotranks!(hssA::HssMatrix, co, ro, cticks, rticks, level, cl)
  if isleaf(hssA)
    m, n = size(hssA)
    plot!(_rectangle(n,m,co,ro), color=:tomato, label=false)
  else
    m1, n1 = hssA.sz1
    m2, n2 = hssA.sz2
    # plot diagonal blocks and ticks
    _plotranks!(hssA.A11, co, ro, cticks, rticks, level+1, cl)
    if level < cl
      append!(rticks, ro+m1)
      append!(cticks, co+n1)
    end
    _plotranks!(hssA.A22, co+n1, ro+m1, cticks, rticks, level+1, cl)
    # plot off-diagonal blocks
    plot!(_rectangle(n2, m1, co+n1, ro), color=:aliceblue, label=false)
    if level < cl annotate!((co+n1+0.5*n2, ro+0.5*m1, text(max(size(hssA.B12)...), 8))) end
    plot!(_rectangle(n1, m2, co, ro+m1), color=:aliceblue, label=false)
    if level < cl annotate!((co+0.5*n1, ro+m1+0.5*m2, text(max(size(hssA.B21)...), 8))) end
  end
  return rticks, cticks
end

# convenience function imitating the behavior of pcolor in matlab
pcolor(A) = heatmap(A, xlims=(1,size(A,2)), ylims=(1,size(A,1)), yflip=true, aspect_ratio=:equal);
