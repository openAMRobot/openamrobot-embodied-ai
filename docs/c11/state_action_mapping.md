<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# C11 state/action mapping review

The current versioned mapping is
[robot-profile-compatibility-v1.md](../c10/robot-profile-compatibility-v1.md).
It lists every C9 16-channel and historical C11 19-channel field, distinguishes
measured observations from commands, identifies units/sources/frames and marks
unapproved physical mappings/limits UNCONFIRMED.

The previous version of this document at PR #4 head `db199924` incorrectly
assigned ROS-like names/radians/metre grippers to the C9 adapter profile and
called the historical 19-channel list two seven-joint arms plus grippers. C9's
adapter representation uses degrees for all sixteen entries, including grippers;
the historical nineteen-entry list has six arm joints plus one gripper per side.
No implicit conversion or mapping between them is approved.

No lift action channel or physical acceptance follows from a successful training
run. The newer combined lift model remains a separate configuration requiring
owner-reviewed observation mapping. JSON/Markdown execution artifacts retain
NOT_VERIFIED_PHYSICAL and production NOT_VERIFIED.
