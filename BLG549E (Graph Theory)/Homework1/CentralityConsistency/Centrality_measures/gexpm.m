function Y = gexpm(D,tol,m1)
%GEXPM Generalized matrix exponential.
%   This function numerically solves first-order, linear, homogeneous
%   differential equation systems, with non-constant coefficients, by
%   generalization of the Pade-approximant method for exponential matrices.
%   The equations are described in matrix form as
%     Y'(t) = D(t)*Y(t)
%   where D and Y are square-matrix functions of scalar t. The initial
%   condition is Y(0) = I (the identity matrix), and the result is Y(1).
%   For the special case of a constant coefficient matrix D, gexpm is
%   equivalent to the standard matrix exponential (expm).
%
% Syntax:
%   Y = gexpm(D);
%   Y = gexpm(D,tol);
%   Y = gexpm(D,tol,m1);
%
% Inputs:
%
%   D: either (1) a scalar or square matrix (real or complex), or (2) a
%   function handle, where D(t) returns a scalar or square matrix for
%   scalar real argument t, 0<=t<=1.
%   * Derivative coefficient matrix (constant or non-constant).
%
%   tol: positive real scalar or [], optional, default is [].
%   * target accuracy tolerance (relative to |Y-I|). The default [] is
%   converted to tol = eps.
%
%   m1: true or false, optional, default is false.
%   * "minus 1" flag: if m1 = false the generalized exponential is Y; if
%   true it is Y+I. gexpm is analogous to the expm1 function
%   ("exponential-minus-1") when m1 is true.
%
% Output:
%
%   Y: scalar or square matrix (real or complex), size-matched to D (or
%   D(t)).
%   * If D is numeric, then Y = expm(D). If D is a function handle, then Y
%   is the solution to Y'(t) = D(t)*Y(t) at t = 1, with Y(0) = I.
%
% Notes:
%
% For both cases of constant and non-constant D, the solution is determined
% from a Pade approximation of order 6, using scale-and-square for constant
% D and multi-step integration for non-constant D. The form of the Pade
% approximant is outlined in the associated document
% KJohnson_2015_04_01.pdf. Notes on error control are in the code comments.
%

% Version 04/06/2015
%
% Copyright (c) 2015, Kenneth C. Johnson
%   (KJ Innovation  kjinnovation@earthlink.net)
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if nargin<2 || isequal(tol,[])
    tol = eps;
elseif ~(isscalar(tol) && isfloat(tol) && isreal(tol) && tol>0)
    error('gexpm:error','Invalid tol (type or size).')
end

if nargin<3
    m1 = false;
elseif ~(isequal(m1,true) || isequal(m1,false))
    error('gexpm:error','Invalid m1 (must be true or false).')
end

if ~isa(D,'function_handle')
    if ~(isfloat(D) && ismatrix(D) && diff(size(D))==0)
        error('gexpm:error','Invalid D (type or size).')
    end
    
    % Y = expm(D).
    if isempty(D) || all(D(:)==0)
        if m1
            Y = zeros(size(D));
        else
            Y = eye(size(D));
        end
        return
    end
    
    % Calculate Y = expm(D), Ym1 = Y-I.
    %
    % Scale and square: Y = expm(D/n)^n; n = 2^s (non-negative integer s)
    %
    % Pade approximation: expm(D/n) = R
    %   = (I-Dh+(2/5)*Dh^2-(1/15)*Dh^3)\(I+Dh+(2/5)*Dh^2+(1/15)*Dh^3)
    % where Dh = D*h; h = 1/(2*n) (h = integration half-step)
    %
    % Pade approximation error: R-expm(D/n) = (-2/1575)*(Dh)^7
    %
    % Set Y = R^n. Approximate absolute error:
    %   Y-expm(D) = R^n-expm(D/n)^n = n*R^(n-1)*(R-expm(D/n))
    %     = n*Y*R^(-1)*(-2/1575)*(Dh)^7
    % R^(-1) is close to I, so
    %   Y-expm(D) = n*Y*(-2/1575)*(Dh)^7 (approx)
    %
    % Large D (|D|>=1): Bound the relative error magnitude by tol,
    %   n*(2/1575)*|(Dh)^7| <= tol
    %
    % Small D (|D|<1): |Y-I| is of order 1 or less; the absolute error
    % Y-expm(D) is of order n*(-2/1575)*(Dh)^7. Bound the error magnitude
    % by tol*|D| to preserve relative accuracy of Ym1:
    %   n*(2/1575)*|(Dh)^7| <= tol*|D|
    %
    % Combine large/small Dh conditions conjunctively:
    %   n*(2/1575)*|(Dh)^7| <= tol*min(1,|D|)
    %
    % Substitute h = 1/(2*n):
    %   (2*n)^6 >= |D^7|/(1575*tol*min(1,|D|))
    % Substitute n = 2^s:
    %   s >= log2(|D^7|/(1575*tol*min(1,|D|)))/6-1
    % (Use the Frobenius norm for |...| to preserve symmetry of expm under
    % matrix transposition.)
    D_2 = D*D;
    D_3 = D_2*D;
    s = ceil(log2(norm(D_3*D_3*D,'fro')/ ...
        (1575*tol*min(1,norm(D,'fro'))))/6-1);
    s = max(s,0);
    % Get Pade approximation for expm(D*h2) = Y =
    %   (I-Dh+(2/5)*Dh^2-(1/15)*Dh^3)\(I+Dh+(2/5)*Dh^2+(1/15)*Dh^3)
    % Ym1 = Y-I =
    %   (I-Dh+(2/5)*Dh^2-(1/15)*Dh^3)\(2*(Dh+(1/15)*Dh^3))
    % (Calculate Ym1, not Y, to avoid precision loss from dominant I
    % terms when Dh is small.)
    h2 = 2^-s;
    h = h2/2;
    D = D*h; % Dh
    D_2 = D_2*(h*h); % Dh^2
    D_3 = D_3*(h*h*h); % Dh^3
    I = eye(size(D));
    Ym1 = D+(1/15)*D_3;
    Ym1 = (I+(2/5)*D_2-Ym1)\(2*Ym1);
    % Y = Ym1+I = expm(D)
    % Square Y (i.e., Y <-- Y*Y) s times to get Y = expm(D*2^s).
    for j = 1:s
        Ym1 = Ym1*Ym1+2*Ym1; % (Ym1+I) <-- (Ym1+I)*(Ym1+I)
    end
    if m1
        Y = Ym1; % without precision loss from I diagonal
    else
        Y = Ym1+I;
    end
    return
end

% Non-constant D.
D0 = D(0);
if ~(isfloat(D0) && ismatrix(D0) && diff(size(D0))==0)
    error('gexpm:error','Invalid D(0) (type or size).')
end

if isempty(D0)
    Y = [];
    return
end

% Search for an initial 3-point t interval in which |D(t)| is monotonic.
% Initialize D0 to the D(t) with largest norm (for the purpose of
% initializing the first integration interval).
t = [0,.5,1];
Dt = {D0,D(t(2)),D(t(3))};
normDt = [norm(Dt{1},'fro'),norm(Dt{2},'fro'),norm(Dt{3},'fro')];
while sign(diff(normDt))==-1
    t = [0,t(2)/2,t(2)];
    Dt = {Dt{1},D(t(2)),Dt{2}};
    normDt = [normDt(1),norm(Dt{2},'fro'),normDt(2)];
end
if max(normDt)==0
    h2 = 1;
else
    if normDt(1)>normDt(3)
        D0 = Dt{1};
    else
        D0 = Dt{3};
    end    
    % Initialize step size (h2) using constant-D formula for single-step
    % integration error:
    %   Y(t+h2) = R*Y(t); R error = (-2/1575)*(Dh)^7
    % (Dh = D0*h, h = h2/2). Include a factor of 1/h2 to estimate the
    % cumulative error; bound the cumulative error by tol for large D
    % (|D0|>=1) or by tol*|D| for small D (|D0|<1),
    %   (1/h2)*(2/1575)*|(D0*h)^7| <= tol*min(1,|D0|)
    % Make this an equality; solve for h:
    %   h = (1575*tol*min(1,|D0|)/|D0^7|)^(1/6)
    D0_2 = D0*D0;
    D0_3 = D0_2*D0;
    h2 = 2*exp(log((1575*tol*min(1,norm(D0,'fro'))/ ...
        (norm(D0_3*D0_3*D0,'fro'))))/6);
    if h2>1
        h2 = 1;
    end
end
clear D0 D0_2 D0_3
% t(1:5): D(t) evaluation points for current integration interval
t = h2*[0,.25,.5,.75,1];
Dt = {D(t(1)),D(t(2)),D(t(3)),D(t(4)),D(t(5))};
DDt = {Dt{1}*Dt{1},[],[],[],Dt{5}*Dt{5}}; % D^2 at t(1) and t(5)
Ym1 = zeros(size(Dt{1})); % Y-I at t(1)
while true
    % Y(t(end)) = R*Y(t(1)); use Pade approximant for R:
    Rm1 = calc_Rm1(t,Dt,DDt); % Rm1 = R-I
    abs_Rm1_error = inf; % for checking convergence
    count = 0;
    while true
        count = count+1;
        % Subdivide the t sampling; recompute Rm1 with h reduced by half.
        % (The result, Rm1_, will be combined with the preceding Rm1 via
        % Richardson extrapolation to estimate and correct the Pade
        % approximation error.)
        t = [t(1),reshape([0.5*(t(1:end-1)+t(2:end));t(2:end)], ...
            [1,2*(length(t)-1)])];
        Dt = [Dt(1),reshape([cell(1,length(Dt)-1);Dt(2:end)], ...
            [1,2*(length(Dt)-1)])];
        for j = 2:2:length(t)-1
            Dt{j} = D(t(j));
        end
        DDt = [DDt(1),reshape([cell(1,length(DDt)-1);DDt(2:end)], ...
            [1,2*(length(DDt)-1)])];
        for j = 5:8:length(t)-4
            DDt{j} = Dt{j}*Dt{j};
        end
        Rm1_ = calc_Rm1(t(1:5),Dt(1:5),DDt(1:5));
        for j = 5:4:length(t)-4
            Rm1__ = calc_Rm1(t(j:j+4),Dt(j:j+4),DDt(j:j+4));
            Rm1_ = Rm1__*Rm1_+Rm1__+Rm1_; % (Rm1_+I) <-- (Rm1__+I)*(Rm1_+I)
        end
        % Richardson extrapolation: The Rm1_ single-step error (from t(j)
        % to t(j+4)) is proportional to h^7 (h = t(3)-t(1)); the cumulative
        % error (from t(1) to t(end)) is proportional to h^6. Denote by
        % Rm1_extrap the extrapolation of Rm1 to zero h.
        %   Rm1_ = Rm1_extrap+c*(t(3)-t(1))^6
        %     = Rm1_extrap+c*((t(5)-t(1))/2)^6  (c unknown)
        %   Rm1 = Rm1_extrap+c*(t(5)-t(1))^6
        % --> (Rm1-Rm1_extrap) = (Rm1_-Rm1_extrap)*2^6,
        %   Rm1_extrap = Rm1_-(Rm1-Rm1_)/(2^6-1) = Rm1_-Rm1_error
        Rm1_error = (Rm1-Rm1_)/63;
        abs_Rm1_error_ = norm(Rm1_error,'fro');
        % Error target (with factor (t(end)-t(1)) representing the portion
        % of the tolerance budget allocated to the current integration
        % interval):
        %   |Rm1_error| <= tol*(t(end)-t(1))
        if abs_Rm1_error_ <= (t(end)-t(1))*tol
            Rm1 = Rm1_-Rm1_error; % Richardson extrapolation
            break
        end
        if abs_Rm1_error_ > 0.5*abs_Rm1_error
            % Convergence rate should be faster, is limited by numeric
            % precision.
            error('gexpm:error','Unable to achieve specified tolerance.')
        end
        % Set up for next loop:
        Rm1 = Rm1_;
        abs_Rm1_error = abs_Rm1_error_;
    end
    Ym1 = Rm1*Ym1+Rm1+Ym1; % (Ym1+I) <-- (Rm1+I)*(Ym1+I)
    % Ym1 = Y-I at t(end).
    if t(end)==1
        break
    end
    h2 = t(9)-t(1); % integration step for last Rm1
    if count==1
        % Accuracy target was achieved on first iteration of t subdivision
        % loop. The step size may be too small; double it.
        h2 = h2*2;
    end
    if t(end)+h2<1
        t = t(end)+h2*[0,.25,.5,.75,1];
    else
        h2 = 1-t(end);
        t = t(end)+h2*[0,.25,.5,.75,1];
        t(end) = 1; % exact; avoid precision error
    end
    Dt = {Dt{end},D(t(2)),D(t(3)),D(t(4)),D(t(5))};
    DDt = {DDt{end},[],[],[],Dt{5}*Dt{5}};
end
if m1
    Y = Ym1;
else
    Y = Ym1+eye(size(Ym1));
end
end

function Rm1 = calc_Rm1(t,Dt,DDt)
% Calculate Rm1 = R-I where R is the Pade ratio Q\P, which is used to
% estimate Y(t(end)) = R*Y(t(1)). D(t) is sampled at t(1) ... t(5)
% (corresponding to Dt{1} ... Dt{5}); D(t)^2 is sampled only at t(1) and
% t(5) (corresponding to DDt{1}, DDt{5}).
%
% Calculate Rm1 (not R) to avoid precision degredation from dominant I
% diagonal.
%
%  Pm1 = P-I, Qm1 = Q-I, Rm1 = R-I,
%    R = Q\P --> Rm1 = (Qm1+I)\(Pm1+I)-I = (Qm1+I)\(Pm1-Qm1)
h = t(5)-t(3); % integration half-step
hh = h*h;
Qm1 = (-(h*2/45)*Dt{2}-(h*2/15)*Dt{3}-(h*2/3)*Dt{4} ...
    -(h*7/45)*Dt{5}) ...
    +((hh/15)*Dt{2}+(hh/5)*Dt{3}+(hh*11/15)*Dt{4})* ...
    (((2/45)*Dt{2}-(1/5)*Dt{3}+(2/5)*Dt{4}+(7/45)*Dt{5}) ...
    -(h/15)*DDt{5});
Pm1 = ((h*2/45)*Dt{4}+(h*2/15)*Dt{3}+(h*2/3)*Dt{2} ...
    +(h*7/45)*Dt{1}) ...
    +((hh/15)*Dt{4}+(hh/5)*Dt{3}+(hh*11/15)*Dt{2})* ...
    (((2/45)*Dt{4}-(1/5)*Dt{3}+(2/5)*Dt{2}+(7/45)*Dt{1}) ...
    +(h/15)*DDt{1});
Rm1 = (Qm1+eye(size(Qm1)))\(Pm1-Qm1);
end
