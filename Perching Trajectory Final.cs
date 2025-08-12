using UnityEngine;

/// <summary>
/// Perching trajectory controller with five sequential phases:
/// 1) Fly-to-Start (to the tau-trajectory start point)
/// 2) Pause-at-Start (hold position for a few seconds)
/// 3) Tau-Law Approach (curved approach with α-coupling)
/// 4) Flat Phase (horizontal, constant-altitude motion)
/// 5) Incline Phase (ascend along an incline after flat)
///
/// Coordinate system (Unity): X-Z horizontal plane, Y up.
/// Targets:
///  - targetA: SAM head
///  - targetB: Buoy
/// </summary>
public class PerchingTrajectoryFinal : MonoBehaviour
{
    [Header("Targets (assign both)")]
    public Transform targetA; // SAM head
    public Transform targetB; // Buoy

    [Header("Start/Touchdown Geometry")]
    [Tooltip("Horizontal offset from midpoint (perpendicular to A→B on XZ plane) where tau-trajectory starts (meters). Negative flips side.")]
    public float horizontalDistanceFromTarget = -10f;
    [Tooltip("Vertical offset above midpoint where tau-trajectory starts (meters).")]
    public float verticalDistanceFromTarget = 7f;
    [Tooltip("Final touchdown height above midpoint (meters).")]
    public float verticalHeightOffset = 0.3f;

    [Header("Tau Trajectory Parameters")]
    [Tooltip("Number of discrete waypoints in the tau trajectory.")]
    public int numSteps = 400;
    [Tooltip("Time between waypoints (s) – controls the publish/update rate for tau steps.")]
    public float waypointDt = 0.05f;
    [Tooltip("Initial velocity used in tau timing (m/s).")]
    public float initialVelocity = 5.0f;
    [Tooltip("Tau shape parameter k (0<k<1 typical).")]
    public float tauShapeParam = 0.4f;
    [Tooltip("α-coupling exponent (kd alpha).")]
    public float kdAlpha = 0.8f;

    [Header("Phase Thresholds & Speeds")]
    [Tooltip("Speed (m/s) when flying to the tau start point.")]
    public float flyToStartVelocity = 5.0f;
    [Tooltip("Within this distance (m) from start we consider 'arrived' and begin pause-at-start.")]
    public float tauTrajectoryStartThreshold = 0.2f;
    [Tooltip("Distance (m) from touchdown at which we switch from tau to flat phase.")]
    public float ropeProximityThreshold = 1.0f;
    [Tooltip("Flat phase forward distance (m).")]
    public float flatForwardDistance = 4.0f;
    [Tooltip("Flat and incline phase speed (m/s).")]
    public float flatForwardVelocity = 1.0f;
    [Tooltip("Incline phase total distance (m).")]
    public float inclineDistance = 10.0f;
    [Tooltip("Incline angle in degrees (relative to horizontal).")]
    public float inclineAngleDegrees = 30f;

    [Header("Pauses")]
    [Tooltip("Seconds to pause after reaching the start point before beginning tau phase.")]
    public float pauseAtStartDuration = 5f;

    // Pause-at-start state
    private bool pauseAtStart = false;
    private float pauseStartTime = 0f;

    // --- Computed geometry ---
    private Vector3 startPoint;      // tau start
    private Vector3 touchdownPoint;  // midpoint + verticalHeightOffset

    // --- Tau trajectory ---
    private Vector3[] tauWaypoints;
    private int tauIndex = 0;
    private float tauTimeAcc = 0f;

    // --- Phase state ---
    private bool computedGeometry = false;
    private bool reachedStart = false;
    private bool tauPhase = false;
    private bool flatPhase = false;
    private bool inclinePhase = false;
    private bool finished = false;

    // --- Flat phase state ---
    private Vector3 flatDirection;     // horizontal unit vector (XZ)
    private float flatTraveled = 0f;
    private Vector3 flatBasePoint;     // anchor at switch from tau→flat

    // --- Incline phase state ---
    private Vector3 inclineDirection;  // 3D unit vector
    private float inclineTraveled = 0f;

    private void Start()
    {
        if (targetA == null || targetB == null)
        {
            Debug.LogError("[PerchingTrajectoryFinal] Assign both targetA (SAM) and targetB (Buoy).");
            enabled = false;
            return;
        }

        ComputeStartAndTouchdown();
    }

    private void FixedUpdate()
    {
        if (finished) return;

        if (!computedGeometry)
        {
            ComputeStartAndTouchdown();
            if (!computedGeometry) return;
        }

        // Phase 1: Fly-to-Start
        if (!reachedStart)
        {
            MoveTowardStart();
            return;
        }

        // Phase 2: Pause-at-Start
        if (pauseAtStart)
        {
            HandlePauseAtStart();
            return;
        }

        // Phase 3: Tau-Law Approach (until rope proximity)
        if (tauPhase)
        {
            RunTauPhase();
            return;
        }

        // Phase 4: Flat Phase (constant altitude, horizontal)
        if (flatPhase && !inclinePhase)
        {
            RunFlatPhase();
            return;
        }

        // Phase 5: Incline Phase
        if (inclinePhase)
        {
            RunInclinePhase();
            return;
        }

        // All done
        finished = true;
        Debug.Log("[PerchingTrajectoryFinal] All phases completed.");
    }

    // --------------------------
    // Geometry & Trajectory
    // --------------------------
    private void ComputeStartAndTouchdown()
    {
        // Midpoint of A & B in 3D
        Vector3 pA = targetA.position;
        Vector3 pB = targetB.position;
        Vector3 mid3D = (pA + pB) * 0.5f;

        // Perpendicular direction on XZ plane to A→B
        Vector2 A2 = new Vector2(pA.x, pA.z);
        Vector2 B2 = new Vector2(pB.x, pB.z);
        Vector2 delta = B2 - A2;
        if (delta.sqrMagnitude < 1e-8f)
        {
            Debug.LogError("[PerchingTrajectoryFinal] targetA and targetB are coincident in XZ.");
            return;
        }
        Vector2 perp = new Vector2(-delta.y, delta.x).normalized;

        // Start is offset horizontally along perp and vertically above midpoint
        Vector2 start2D = new Vector2(mid3D.x, mid3D.z) + perp * horizontalDistanceFromTarget;
        startPoint = new Vector3(start2D.x, mid3D.y + verticalDistanceFromTarget, start2D.y);

        // Touchdown is midpoint + verticalHeightOffset
        touchdownPoint = mid3D + Vector3.up * verticalHeightOffset;

        computedGeometry = true;
        reachedStart = false;
        tauPhase = false;
        flatPhase = false;
        inclinePhase = false;
        finished = false;

        // Precompute tau waypoints (but don't start using them until we begin tau)
        tauWaypoints = GenerateTauTrajectoryAlphaCoupled(startPoint, touchdownPoint, numSteps, initialVelocity, tauShapeParam, kdAlpha);
        tauIndex = 0;
        tauTimeAcc = 0f;

        Debug.Log($"[PerchingTrajectoryFinal] Geometry ready.\nStart: {startPoint}\nTouchdown: {touchdownPoint}\nTau steps: {numSteps}");
    }

    /// <summary>
    /// Tau-law with α-coupling (matches the Python logic):
    /// - Compute delta = p0 - p_td
    /// - d0 = |delta|
    /// - dir_xz = horizontal direction from touchdown toward start (XZ normalized)
    /// - alpha0 = arcsin((p0.y - p_td.y)/d0)
    /// - tau0 = -d0 / v0 ; t_d = -tau0 / k
    /// - For i: d = d0 * (1 - t/t_d)^(1/k)
    /// - α = α0 * (d/d0)^(1/kd)
    /// - h = d cos α (horizontal reach), z = d sin α (vertical rise)
    /// - point = p_td + dir_xz * h + (0, z, 0)
    /// </summary>
    private Vector3[] GenerateTauTrajectoryAlphaCoupled(
        Vector3 p0, Vector3 p_td, int steps, float v0, float k, float kdAlphaExp)
    {
        Vector3[] traj = new Vector3[Mathf.Max(steps, 2)];

        Vector3 delta = p0 - p_td;
        float d0 = delta.magnitude;
        if (d0 < 1e-6f)
        {
            for (int i = 0; i < traj.Length; i++) traj[i] = p_td;
            return traj;
        }

        // Horizontal (XZ) direction
        Vector2 deltaXZ = new Vector2(delta.x, delta.z);
        Vector2 dirXZ = deltaXZ.normalized;

        // Initial alpha (pitch from horizontal to delta): use arcsin(vertical/distance)
        float alpha0 = Mathf.Asin(Mathf.Clamp((p0.y - p_td.y) / d0, -1f, 1f));

        // Tau timing
        float tau0 = -d0 / Mathf.Max(1e-6f, v0);
        float t_d = -tau0 / Mathf.Max(1e-6f, k);
        float inv_k = 1f / Mathf.Max(1e-6f, k);
        float inv_kd = 1f / Mathf.Max(1e-6f, kdAlphaExp);

        for (int i = 0; i < traj.Length; i++)
        {
            float t = t_d * i / (traj.Length - 1);
            float d = d0 * Mathf.Pow(1f - t / t_d, inv_k);

            float alpha = alpha0 * Mathf.Pow(d / d0, inv_kd);
            float cosA = Mathf.Cos(alpha);
            float sinA = Mathf.Sin(alpha);

            float h = d * cosA; // horizontal
            float z = d * sinA; // vertical

            Vector3 horizontal = new Vector3(dirXZ.x * h, 0f, dirXZ.y * h);
            traj[i] = p_td + horizontal + new Vector3(0f, z, 0f);
        }

        return traj;
    }

    // --------------------------
    // Phase Implementations
    // --------------------------
    private void MoveTowardStart()
    {
        Vector3 pos = transform.position;
        Vector3 toStart = startPoint - pos;
        float dist = toStart.magnitude;

        if (!pauseAtStart && dist <= tauTrajectoryStartThreshold)
        {
            // Reached the start position → begin pause timer
            reachedStart = true;
            pauseAtStart = true;
            pauseStartTime = Time.time;
            transform.position = startPoint; // snap to start
            Debug.Log($"[PerchingTrajectoryFinal] Reached start → pausing for {pauseAtStartDuration} seconds.");
            return;
        }

        // Still moving toward start
        float step = flyToStartVelocity * Time.fixedDeltaTime;
        Vector3 next = dist > 1e-6f ? pos + toStart.normalized * Mathf.Min(step, dist) : startPoint;
        transform.position = next;
        FaceVelocity(toStart);
    }

    private void HandlePauseAtStart()
    {
        // Hold exact start position (optional)
        transform.position = startPoint;

        if (Time.time - pauseStartTime >= pauseAtStartDuration)
        {
            pauseAtStart = false;
            tauPhase = true;
            tauIndex = 0;
            tauTimeAcc = 0f;
            Debug.Log("[PerchingTrajectoryFinal] Pause complete → begin tau phase.");
        }
    }

    private void RunTauPhase()
    {
        if (tauWaypoints == null || tauWaypoints.Length == 0)
        {
            // Fail-safe: jump to touchdown and switch to flat
            transform.position = touchdownPoint;
            SwitchToFlatFrom(touchdownPoint, Vector3.forward);
            return;
        }

        // Advance along tau waypoints at rate waypointDt
        tauTimeAcc += Time.fixedDeltaTime;
        while (tauTimeAcc >= waypointDt && tauIndex < tauWaypoints.Length - 1)
        {
            tauIndex++;
            tauTimeAcc -= waypointDt;
        }

        Vector3 current = tauWaypoints[tauIndex];
        transform.position = current;

        // Check proximity to touchdown to switch to flat
        float remaining = Vector3.Distance(current, tauWaypoints[tauWaypoints.Length - 1]);
        if (remaining <= ropeProximityThreshold || tauIndex >= tauWaypoints.Length - 1)
        {
            // Compute flat direction from last tau step
            Vector3 prev = tauWaypoints[Mathf.Max(0, tauIndex - 1)];
            Vector3 flatVec = current - prev;
            flatVec.y = 0f;
            if (flatVec.sqrMagnitude < 1e-10f)
            {
                // If degenerate, default to A→B horizontal direction
                Vector3 AB = (targetB.position - targetA.position);
                flatVec = new Vector3(AB.x, 0f, AB.z);
            }
            Vector3 flatDir = flatVec.sqrMagnitude > 1e-10f ? flatVec.normalized : Vector3.forward;

            SwitchToFlatFrom(current, flatDir);
        }
        else
        {
            // (Optional) rotate to face direction of travel
            Vector3 vel = (tauIndex + 1 < tauWaypoints.Length)
                ? (tauWaypoints[tauIndex + 1] - current)
                : (current - tauWaypoints[Mathf.Max(0, tauIndex - 1)]);
            FaceVelocity(vel);
        }
    }

    private void SwitchToFlatFrom(Vector3 basePoint, Vector3 directionXZ)
    {
        flatPhase = true;
        tauPhase = false;
        inclinePhase = false;

        flatBasePoint = basePoint;
        directionXZ.y = 0f;
        flatDirection = directionXZ.sqrMagnitude > 1e-10f ? directionXZ.normalized : Vector3.forward;
        flatTraveled = 0f;

        Debug.Log($"[PerchingTrajectoryFinal] Switching to flat phase. Flat dir: {flatDirection}, base: {flatBasePoint}");
    }

    private void RunFlatPhase()
    {
        if (flatTraveled < flatForwardDistance)
        {
            float stepLen = flatForwardVelocity * Time.fixedDeltaTime;
            flatTraveled = Mathf.Min(flatTraveled + stepLen, flatForwardDistance);

            Vector3 next = flatBasePoint + flatDirection * flatTraveled;
            next.y = flatBasePoint.y; // keep altitude constant in flat phase
            Vector3 vel = next - transform.position;

            transform.position = next;
            FaceVelocity(vel);
            return;
        }

        // Initialize incline once
        if (!inclinePhase)
        {
            float angleRad = Mathf.Deg2Rad * inclineAngleDegrees;
            float h = Mathf.Cos(angleRad);
            float v = Mathf.Sin(angleRad);

            // Incline direction is combination of horizontal (flatDirection) and vertical up
            Vector3 horiz = flatDirection * h;
            Vector3 vec3D = new Vector3(horiz.x, v, horiz.z);
            inclineDirection = vec3D.sqrMagnitude > 1e-10f ? vec3D.normalized : Vector3.up;

            inclineTraveled = 0f;
            inclinePhase = true;

            Debug.Log("[PerchingTrajectoryFinal] Starting incline phase.");
        }
    }

    private void RunInclinePhase()
    {
        if (inclineTraveled < inclineDistance)
        {
            float stepLen = flatForwardVelocity * Time.fixedDeltaTime;
            inclineTraveled = Mathf.Min(inclineTraveled + stepLen, inclineDistance);

            // Start point: end of flat motion
            Vector3 startOfIncline = flatBasePoint + flatDirection * flatForwardDistance;
            Vector3 next = startOfIncline + inclineDirection * inclineTraveled;

            Vector3 vel = next - transform.position;
            transform.position = next;
            FaceVelocity(vel);
            return;
        }

        // Completed all phases
        inclinePhase = false;
        finished = true;
        Debug.Log("[PerchingTrajectoryFinal] Incline complete. Sequence finished.");
    }

    // --------------------------
    // Helpers
    // --------------------------
    private void FaceVelocity(Vector3 vel)
    {
        vel.y = 0f; // yaw-only facing (optional)
        if (vel.sqrMagnitude > 1e-8f)
        {
            Quaternion look = Quaternion.LookRotation(vel.normalized, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, look, 0.2f);
        }
    }

#if UNITY_EDITOR
    // Optional: visualize key points in the editor
    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.cyan;
        Gizmos.DrawSphere(startPoint, 0.2f);

        Gizmos.color = Color.green;
        Gizmos.DrawSphere(touchdownPoint, 0.2f);

        if (tauWaypoints != null && tauWaypoints.Length > 1)
        {
            Gizmos.color = Color.yellow;
            for (int i = 0; i < tauWaypoints.Length - 1; i++)
            {
                Gizmos.DrawLine(tauWaypoints[i], tauWaypoints[i + 1]);
            }
        }

        if (flatPhase)
        {
            Gizmos.color = Color.magenta;
            Vector3 endFlat = flatBasePoint + flatDirection * flatForwardDistance;
            Gizmos.DrawLine(flatBasePoint, endFlat);
        }

        if (inclinePhase)
        {
            Gizmos.color = Color.red;
            Vector3 startIncline = flatBasePoint + flatDirection * flatForwardDistance;
            Vector3 endIncline = startIncline + inclineDirection * inclineDistance;
            Gizmos.DrawLine(startIncline, endIncline);
        }
    }
#endif
}
